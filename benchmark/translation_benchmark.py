#!/usr/bin/env python3

import argparse
import csv
import json
import os
import random
import socket
import time
import urllib.request
import statistics
import threading
import re
import codecs
import requests
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from translate import (
    translate,
    add_translate_args,
    config_from_args,
    TimingInfo,
    TranslateConfig
)


@dataclass
class BenchmarkResult:
    chunk_id: int
    chunk_text: str
    duration: float
    success: bool
    error: Optional[str] = None
    translation: Optional[str] = None
    timed_out: bool = False
    completed_at: float = 0.0
    pending_time: float = 0.0  # Time from submission to HTTP start
    timing: Optional[TimingInfo] = None


def download_war_and_peace() -> str:
    url = "https://www.gutenberg.org/files/2600/2600-0.txt"
    print(f"Downloading War and Peace from {url}...")
    with urllib.request.urlopen(url, timeout=30) as response:
        text = response.read().decode('utf-8')
    print(f"Downloaded {len(text)} characters")
    return text


def parse_log_file(logfile: str) -> List[tuple]:
    """
    Parse log file and extract (target_lang, texts) tuples.
    Returns list of (target_lang, [text1, text2, ...]) tuples.
    """
    queries = []
    with open(logfile, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.search(r'<HTTP_REQUEST[^>]*>(.*?)</HTTP_REQUEST>', line)
            if not match:
                continue

            payload_escaped = match.group(1)

            try:
                # Decode escape sequences as bytes, then decode UTF-8
                payload_bytes = codecs.decode(payload_escaped, 'unicode_escape')
                if isinstance(payload_bytes, str):
                    # Python 3: unicode_escape returns str, need to re-encode as latin1 to get bytes
                    payload_bytes = payload_bytes.encode('latin1')
                payload_json_str = payload_bytes.decode('utf-8')
                data = json.loads(payload_json_str)
            except (json.JSONDecodeError, UnicodeDecodeError, UnicodeEncodeError):
                continue

            if 'messages' not in data:
                continue

            user_content = None
            for msg in data['messages']:
                if msg.get('role') == 'user':
                    user_content = msg.get('content', '')
                    break

            if not user_content:
                continue

            try:
                user_data = json.loads(user_content)
            except json.JSONDecodeError:
                continue

            target_lang = user_data.get('target_lang', 'Unknown')
            texts = [item.get('text', '') for item in user_data.get('texts', [])]

            if texts:
                queries.append((target_lang, texts))

    return queries


def split_into_chunks(text: str, chunk_length: int, single_query: bool = False) -> List[str]:
    if single_query:
        return [text.strip()]

    text = text.strip()
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1
        if current_length + word_length > chunk_length and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def calculate_timing_stats(results: List[BenchmarkResult]) -> Optional[dict]:
    """Calculate timing statistics from results with timing info.
    
    Returns dict with worker_overheads, proxy_overheads, client_overheads, network_overheads
    or None if no valid timing data.
    """
    results_with_timing = [r for r in results if r.timing]
    
    if not results_with_timing:
        return None
    
    worker_overheads = []
    proxy_overheads = []
    client_overheads = []
    network_overheads = []

    for r in results_with_timing:
        client_oh, proxy_oh, worker_oh = r.timing.overheads()
        
        if worker_oh > 0:
            worker_overheads.append(worker_oh)
        if proxy_oh > 0:
            proxy_overheads.append(proxy_oh)
        if client_oh > 0:
            client_overheads.append(client_oh)
        
        # Network overhead = total duration - client duration
        cd = r.timing.client_duration()
        if cd is not None and r.duration > 0:
            net_oh = r.duration - cd
            if net_oh > 0:
                network_overheads.append(net_oh)
    
    # Return None if we have no data at all
    if not worker_overheads and not proxy_overheads and not client_overheads and not network_overheads:
        return None
    
    return {
        'count': len(results_with_timing),
        'worker_overheads': worker_overheads,
        'proxy_overheads': proxy_overheads,
        'client_overheads': client_overheads,
        'network_overheads': network_overheads
    }


def print_timing_breakdown(timing_stats: dict, prefix: str = ""):
    """Print timing breakdown statistics.
    
    Args:
        timing_stats: Dict from calculate_timing_stats()
        prefix: Prefix for indentation (e.g., "    ")
    """
    worker_overheads = timing_stats['worker_overheads']
    proxy_overheads = timing_stats['proxy_overheads']
    client_overheads = timing_stats['client_overheads']
    network_overheads = timing_stats['network_overheads']
    count = timing_stats['count']

    print(f"{prefix}Timing Breakdown ({count} requests with timing headers):")

    if network_overheads:
        print(f"{prefix}  Network overhead: avg: {statistics.mean(network_overheads):.3f}s | "
              f"median: {statistics.median(network_overheads):.3f}s | "
              f"p90: {sorted(network_overheads)[int(len(network_overheads) * 0.90)]:.3f}s")

    if worker_overheads:
        print(f"{prefix}  Worker duration:  avg: {statistics.mean(worker_overheads):.3f}s | "
              f"median: {statistics.median(worker_overheads):.3f}s | "
              f"p90: {sorted(worker_overheads)[int(len(worker_overheads) * 0.90)]:.3f}s")
    else:
        print(f"{prefix}  Worker duration:  N/A (no worker headers)")

    if proxy_overheads:
        print(f"{prefix}  Proxy overhead:   avg: {statistics.mean(proxy_overheads):.3f}s | "
              f"median: {statistics.median(proxy_overheads):.3f}s | "
              f"p90: {sorted(proxy_overheads)[int(len(proxy_overheads) * 0.90)]:.3f}s")
    else:
        print(f"{prefix}  Proxy overhead:   N/A (no proxy headers)")

    if client_overheads:
        print(f"{prefix}  Client overhead:  avg: {statistics.mean(client_overheads):.3f}s | "
              f"median: {statistics.median(client_overheads):.3f}s | "
              f"p90: {sorted(client_overheads)[int(len(client_overheads) * 0.90)]:.3f}s")
    else:
        print(f"{prefix}  Client overhead:  N/A (no client headers)")


def translate_chunk(
        chunk_id: int,
        chunk: str,
        target_lang: str,
        config: TranslateConfig,
        start_time: float,
        active_counter=None,
        submit_time: float = 0.0
) -> BenchmarkResult:
    # Calculate pending time (from submit to now)
    pending_time = time.time() - submit_time if submit_time > 0 else 0.0

    request_start = time.time()

    # Initialize timing variables
    worker_start, worker_end = None, None
    proxy_start, proxy_end = None, None
    client_start, client_end = None, None

    try:
        # Increment active counter before HTTP request
        if active_counter is not None:
            with active_counter['lock']:
                active_counter['count'] += 1

        # Call translate function
        result = translate(chunk, target_lang=target_lang, config=config)

        # Extract translation and timing from TranslationResult
        translation = result.translation
        timing = result.timing

        # Debug: print all HTTP headers
        if config.verbose and result.headers:
            print(f"\n[DEBUG] HTTP Headers for chunk {chunk_id}:")
            for key, value in result.headers.items():
                print(f"  {key}: {value}")
            print()

        # Decrement active counter after HTTP request completes
        if active_counter is not None:
            with active_counter['lock']:
                active_counter['count'] -= 1

        duration = time.time() - request_start

        if not translation or translation.strip() == "" or "error" in str(translation).lower():
            raise Exception(f"Empty or error translation received: {translation}")

        return BenchmarkResult(
            chunk_id=chunk_id,
            chunk_text=chunk[:100] + "..." if len(chunk) > 100 else chunk,
            duration=duration,
            success=True,
            translation=translation[:100] + "..." if len(translation) > 100 else translation,
            completed_at=time.time() - start_time,
            pending_time=pending_time,
            timing=timing
        )

    except Exception as e:
        # Decrement active counter on error too
        if active_counter is not None:
            with active_counter['lock']:
                active_counter['count'] -= 1

        duration = time.time() - request_start
        error_str = str(e)
        timed_out = "timeout" in error_str.lower() or "timed out" in error_str.lower()

        if config.verbose:
            import traceback
            print(f"\n{'!' * 70}")
            print(f"DEBUG: Chunk {chunk_id} failed")
            print(f"Chunk text (first 200 chars): {chunk[:200]}...")
            print(f"Error: {error_str}")
            print(f"Traceback:")
            traceback.print_exc()
            print(f"{'!' * 70}\n")

        return BenchmarkResult(
            chunk_id=chunk_id,
            chunk_text=chunk[:100] + "..." if len(chunk) > 100 else chunk,
            duration=duration,
            success=False,
            error=error_str,
            timed_out=timed_out,
            completed_at=time.time() - start_time,
            pending_time=pending_time,
            timing=None
        )


def run_benchmark(
        chunks: List[str],
        config: TranslateConfig,
        concurrency: int,
        target_lang: str,
        max_chunks: Optional[int] = None,
        stats_interval: int = 10,
        target_langs: Optional[List[str]] = None,
        load_mode: str = "burst",
        qps: Optional[float] = None
) -> List[BenchmarkResult]:
    """Run the benchmark with specified concurrency using threads

    If target_langs is provided, it should be a list with one target language per chunk.
    Otherwise, target_lang is used for all chunks.

    Load modes:
    - burst: send all requests as fast as possible (default)
    - fixed: maintain fixed number of active requests (concurrency)
    - qps: emulate fixed queries per second with proper event distribution
    """

    if max_chunks:
        chunks = chunks[:max_chunks]
        if target_langs:
            target_langs = target_langs[:max_chunks]

    # Use target_langs if provided, otherwise use target_lang for all
    if target_langs is None:
        target_langs = [target_lang] * len(chunks)

    if len(target_langs) != len(chunks):
        raise ValueError(f"target_langs length ({len(target_langs)}) must match chunks length ({len(chunks)})")

    if load_mode == "qps" and qps is None:
        raise ValueError("qps must be specified when load_mode='qps'")

    print(f"\n{'=' * 70}")
    print(f"Starting benchmark:")
    print(f"  Endpoint: {config.endpoint}" + (" (Azure)" if config.use_azure else ""))
    print(f"  Model: {config.model}")
    if len(chunks) == 1:
        print(f"  Mode: Single query (entire file)")
        print(f"  Total characters: {len(chunks[0])}")
    else:
        print(f"  Total chunks: {len(chunks)}")
        if load_mode == "fixed":
            print(f"  Load mode: fixed ({concurrency} active requests)")
        elif load_mode == "qps":
            print(f"  Load mode: QPS ({qps} queries/sec, max workers: {concurrency})")

    # Show target language info
    unique_langs = set(target_langs)
    if len(unique_langs) == 1:
        print(f"  Target language: {target_lang}")
    else:
        print(
            f"  Target languages: {len(unique_langs)} different ({', '.join(list(unique_langs)[:5])}{'...' if len(unique_langs) > 5 else ''})")

    print(f"  Prompt format: {config.prompt_format}")
    print(f"  Timeout: {config.timeout}s")
    if len(chunks) > 1:
        print(f"  Stats interval: every {stats_interval} requests")
    print(f"{'=' * 70}\n")

    results = []
    results_lock = threading.Lock()
    active_counter = {'count': 0, 'lock': threading.Lock()}
    processing_counter = {'count': 0, 'lock': threading.Lock()}
    submitted_counter = {'count': 0, 'lock': threading.Lock()}  # For QPS mode
    start_time = time.time()

    # Helper functions to hide lock management
    def get_active():
        with active_counter['lock']:
            return active_counter['count']

    def get_processing():
        with processing_counter['lock']:
            return processing_counter['count']

    def inc_processing():
        with processing_counter['lock']:
            processing_counter['count'] += 1

    def dec_processing():
        with processing_counter['lock']:
            processing_counter['count'] -= 1

    def get_completed():
        with results_lock:
            return len(results)

    def get_submitted():
        with submitted_counter['lock']:
            return submitted_counter['count']

    def inc_submitted():
        with submitted_counter['lock']:
            submitted_counter['count'] += 1

    def get_stats(use_submitted=False):
        """Get all stats in one atomic operation"""
        active = get_active()
        processing = get_processing()
        completed = get_completed()
        if use_submitted:
            # QPS mode:
            # - processing = currently executing (after sleep, doing actual work)
            # - queue = submitted to executor but not yet executing
            submitted = get_submitted()
            queue_size = submitted - completed - processing
            # If queue is negative (race condition), set to 0
            queue_size = max(0, queue_size)
        else:
            # Fixed mode: queue = total - completed - processing
            queue_size = len(chunks) - completed - processing
        return active, processing, completed, queue_size

    def process_chunk(chunk_id, chunk, use_submitted=False, submit_time=0.0):
        """Process a single chunk"""
        chunk = chunk.replace('RANDOM_SALT', str(random.randint(0, 999999)))
        chunk_target_lang = target_langs[chunk_id]
        chunk_len = len(chunk)


        print(f"[{chunk_id + 1}/{len(chunks)}] Processing chunk (length: {chunk_len}, target: {chunk_target_lang})...")
        result = translate_chunk(
            chunk_id, chunk, chunk_target_lang, config, start_time,
            active_counter, submit_time
        )

        speed = chunk_len / result.duration if result.duration > 0 else 0
        active, processing, completed, queue_size = get_stats(use_submitted)

        # Show pending time if it was tracked (QPS mode)
        pending_str = f" | pending: {result.pending_time:.2f}s" if result.pending_time > 0 else ""

        # Format timing info if available
        timing_str = ""
        if result.timing:
            parts = []
            client_oh, proxy_oh, worker_oh = result.timing.overheads()
            
            # Calculate network overhead
            cd = result.timing.client_duration()
            if cd is not None and result.duration > 0:
                net_oh = result.duration - cd
                if net_oh > 0:
                    parts.append(f"N:{net_oh:.3f}s")
            
            if worker_oh > 0:
                parts.append(f"W:{worker_oh:.3f}s")
            if proxy_oh > 0:
                parts.append(f"P:{proxy_oh:.3f}s")
            if client_oh > 0:
                parts.append(f"C:{client_oh:.3f}s")
            
            if parts:
                timing_str = " | " + " ".join(parts)
        
        if result.success:
            if chunk_id < 3 or (chunk_id + 1) % 20 == 0 or True:
                print(
                    f"[{chunk_id + 1}/{len(chunks)}] ✓ {result.duration:.2f}s{pending_str}{timing_str} | {chunk_len} chars | {speed:.0f} chars/s | active: {active} | queue: {queue_size}")
                print(f"  Original: {chunk[:100]}...")
                print(f"  Translation to {chunk_target_lang}: {result.translation}")
                print()
            else:
                print(
                    f"[{chunk_id + 1}/{len(chunks)}] ✓ {result.duration:.2f}s{pending_str}{timing_str} | {chunk_len} chars | {speed:.0f} chars/s | active: {active} | queue: {queue_size}")
        else:
            timeout_marker = " [TIMEOUT]" if result.timed_out else ""
            print(
                f"[{chunk_id + 1}/{len(chunks)}] ✗{timeout_marker} {result.duration:.2f}s{pending_str}{timing_str} | {chunk_len} chars | {speed:.0f} chars/s | active: {active} | queue: {queue_size} | {result.error}")
            if config.verbose:
                print(f"  Chunk preview: {chunk[:200]}...")

        # Add result and check if we should print stats
        with results_lock:
            results.append(result)
            should_print = len(results) % stats_interval == 0

        if should_print:
            active, processing, completed, queue_size = get_stats(use_submitted)
            print_stats(results, time.time() - start_time, len(chunks), chunks, active, queue_size, is_final=False)

        return result

    # Common worker wrapper that tracks processing
    def worker_wrapper(chunk_idx, chunk, use_submitted=False, submit_time=0.0):
        inc_processing()
        try:
            process_chunk(chunk_idx, chunk, use_submitted, submit_time)
        finally:
            dec_processing()

    # Execute based on load mode
    if load_mode == "fixed":
        # Fixed mode: maintain exactly N active requests
        chunk_idx = {'value': 0, 'lock': threading.Lock()}

        def fixed_worker():
            while True:
                with chunk_idx['lock']:
                    i = chunk_idx['value']
                    if i >= len(chunks):
                        break
                    chunk_idx['value'] += 1
                worker_wrapper(i, chunks[i])

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(fixed_worker) for _ in range(concurrency)]
            for future in as_completed(futures):
                future.result()

    elif load_mode == "qps":
        # QPS mode: emit requests at fixed rate with Poisson distribution
        # Generate request times using exponential distribution (Poisson process)
        request_times = []
        current_time = 0
        for _ in range(len(chunks)):
            current_time += random.expovariate(qps)
            request_times.append(current_time)

        # Create schedule with chunk info
        schedule = [(request_times[i], i, chunks[i]) for i in range(len(chunks))]
        schedule.sort(key=lambda x: x[0])

        # Main thread submits tasks to executor at scheduled times
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for scheduled_time, chunk_idx, chunk in schedule:
                # Main thread waits until scheduled time
                wait_time = scheduled_time - (time.time() - start_time)
                if wait_time > 0:
                    time.sleep(wait_time)

                # Increment submitted counter and record submit time
                inc_submitted()
                submit_time = time.time()

                # Submit task to executor (worker executes immediately, no sleep)
                future = executor.submit(worker_wrapper, chunk_idx, chunk, True, submit_time)
                futures.append(future)

            # Wait for all to complete
            for future in as_completed(futures):
                future.result()

    total_duration = time.time() - start_time

    print_stats(results, total_duration, len(chunks), chunks, is_final=True)

    return results


def print_stats(results: List[BenchmarkResult], elapsed_time: float, total_chunks: int, all_chunks: List[str],
                active: int = 0, queue_size: int = 0, is_final: bool = False):
    """Print statistics (interim or final)"""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    timed_out = [r for r in results if r.timed_out]

    # Calculate error percentages
    total_count = len(results)
    failed_count = len(failed)
    error_percent = (failed_count / total_count * 100) if total_count > 0 else 0

    # Calculate error percentage by input chars
    total_input_chars = sum(len(all_chunks[r.chunk_id]) for r in results)
    failed_input_chars = sum(len(all_chunks[r.chunk_id]) for r in failed)
    error_chars_percent = (failed_input_chars / total_input_chars * 100) if total_input_chars > 0 else 0

    # Get results from last 30 seconds
    recent_cutoff = elapsed_time - 30
    recent_successful = [r for r in successful if r.completed_at >= recent_cutoff]

    separator = '=' if is_final else '─'
    title = "FINAL RESULTS" if is_final else f"INTERIM STATS ({len(results)}/{total_chunks} completed, {elapsed_time:.1f}s elapsed)"

    print(f"\n{separator * 70}")
    print(f"{title}:")
    if not is_final:
        print(f"  Active queries: {active}")
        print(f"  Queued: {queue_size}")
    print(f"  Total requests: {len(results)}")
    print(f"  Successful: {len(successful)} | Failed: {len(failed)} | Timed out: {len(timed_out)}")
    print(f"  Error rate: {error_percent:.1f}% (by count) | {error_chars_percent:.1f}% (by input chars)")
    if is_final:
        print(f"  Total time: {elapsed_time:.2f}s")

    if successful:
        durations = [r.duration for r in successful]
        pending_times = [r.pending_time for r in successful if r.pending_time > 0]
        input_chars = sum(len(all_chunks[r.chunk_id]) for r in successful)

        # Calculate overall percentiles
        sorted_durations = sorted(durations)
        p50_idx = int(len(sorted_durations) * 0.50)
        p90_idx = int(len(sorted_durations) * 0.90)
        p99_idx = int(len(sorted_durations) * 0.99)

        print(f"\n  Performance Metrics:")
        print(f"    Throughput: {len(successful) / elapsed_time:.2f} req/s")
        print(f"    Input chars/s: {input_chars / elapsed_time:.0f}")

        print(f"\n  Latency:")
        print(f"    Min: {min(durations):.2f}s | Avg: {statistics.mean(durations):.2f}s | Max: {max(durations):.2f}s")
        print(
            f"    P50: {sorted_durations[p50_idx]:.2f}s | P90: {sorted_durations[p90_idx]:.2f}s | P99: {sorted_durations[p99_idx]:.2f}s")
        if is_final and len(durations) > 1:
            print(f"    Std Dev: {statistics.stdev(durations):.2f}s")

        # Show pending time stats if available (QPS mode)
        if pending_times:
            print(f"\n  Pending Time (submit to HTTP start):")
            print(f"    Avg: {statistics.mean(pending_times):.2f}s")
            if is_final:
                print(f"    Median: {statistics.median(pending_times):.2f}s")
                print(f"    Min: {min(pending_times):.2f}s | Max: {max(pending_times):.2f}s")
                sorted_pending = sorted(pending_times)
                p50_pending = sorted_pending[int(len(sorted_pending) * 0.50)]
                p90_pending = sorted_pending[int(len(sorted_pending) * 0.90)]
                p99_pending = sorted_pending[int(len(sorted_pending) * 0.99)]
                print(f"    P50: {p50_pending:.2f}s | P90: {p90_pending:.2f}s | P99: {p99_pending:.2f}s")

        # Show timing breakdown stats
        timing_stats = calculate_timing_stats(successful)
        if timing_stats:
            print()
            print_timing_breakdown(timing_stats, prefix="  ")

    # Show last 30 seconds stats
    if recent_successful and elapsed_time >= 30:
        recent_durations = [r.duration for r in recent_successful]
        recent_pending_times = [r.pending_time for r in recent_successful if r.pending_time > 0]
        recent_input_chars = sum(len(all_chunks[r.chunk_id]) for r in recent_successful)
        recent_window = min(30, elapsed_time - recent_successful[0].completed_at)

        # Calculate recent percentiles
        sorted_recent = sorted(recent_durations)
        p50_recent = sorted_recent[int(len(sorted_recent) * 0.50)]
        p90_recent = sorted_recent[int(len(sorted_recent) * 0.90)]
        p99_recent = sorted_recent[int(len(sorted_recent) * 0.99)]

        print(f"\n  Last 30s Performance:")
        print(f"    Requests completed: {len(recent_successful)}")
        print(
            f"    Avg latency: {statistics.mean(recent_durations):.2f}s | P50: {p50_recent:.2f}s | P90: {p90_recent:.2f}s | P99: {p99_recent:.2f}s")
        if recent_pending_times:
            print(f"    Avg pending: {statistics.mean(recent_pending_times):.2f}s")
        print(f"    Throughput: {len(recent_successful) / recent_window:.2f} req/s")
        print(f"    Input chars/s: {recent_input_chars / recent_window:.0f}")

        # Show timing breakdown for last 30s
        recent_timing_stats = calculate_timing_stats(recent_successful)
        if recent_timing_stats:
            parts = []
            if recent_timing_stats['network_overheads']:
                net_avg = statistics.mean(recent_timing_stats['network_overheads'])
                parts.append(f"Net: {net_avg:.3f}s")
            if recent_timing_stats['worker_overheads']:
                worker_avg = statistics.mean(recent_timing_stats['worker_overheads'])
                parts.append(f"Worker: {worker_avg:.3f}s")
            if recent_timing_stats['proxy_overheads']:
                proxy_avg = statistics.mean(recent_timing_stats['proxy_overheads'])
                parts.append(f"Proxy OH: {proxy_avg:.3f}s")
            if recent_timing_stats['client_overheads']:
                client_avg = statistics.mean(recent_timing_stats['client_overheads'])
                parts.append(f"Client OH: {client_avg:.3f}s")
            
            if parts:
                print(f"    Timing ({recent_timing_stats['count']} with headers): {' | '.join(parts)}")
    
    # Show error details only in final results
    if is_final and failed:
        print(f"\n  Errors:")
        error_counts = {}
        for r in failed:
            error_key = r.error[:50] if r.error else "Unknown"
            error_counts[error_key] = error_counts.get(error_key, 0) + 1

        for error, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"    {count}x: {error}")

    print(f"{separator * 70}\n")


def save_results_to_csv(
        results: List[BenchmarkResult],
        all_chunks: List[str],
        target_langs: List[str],
        model: str,
        csv_path: str
):
    """Save benchmark results to CSV in the same format as the logging system."""
    
    hostname = socket.gethostname()
    
    # CSV columns matching the log format
    fieldnames = [
        'loc', 'model', 'account', 'date', 'microtime', 'dc_id', 'dlog_tag',
        'layer', 'hostname', 'source', 'event', 'data', 'uldata', 'debug',
        'time', 'uid', 'time_utc'
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, result in enumerate(results):
            chunk_text = all_chunks[result.chunk_id]
            target_lang = target_langs[result.chunk_id] if target_langs else 'unknown'
            
            # Calculate completion timestamp
            completion_time = time.time() - (results[-1].completed_at - result.completed_at) if results else time.time()
            
            # Build debug JSON (the important performance data)
            debug_data = {
                'count': 1,
                'target_lang': target_lang,
                'input_bytes': len(chunk_text.encode('utf-8')),
                'input_chars': len(chunk_text),
                'model': model,
                'account': 'benchmark',
                'prompt_bytes': len(chunk_text.encode('utf-8')),
                'prompt_chars': len(chunk_text),
                'output_bytes': len(result.translation.encode('utf-8')) if result.translation else 0,
                'output_chars': len(result.translation) if result.translation else 0,
                'output_errors': 0 if result.success else 1,
                'time_ms': int(result.duration * 1000)
            }
            
            # Add error info if failed
            if not result.success and result.error:
                error_key = f'output_errors_{result.error[:50]}'
                debug_data[error_key] = 1
            
            # Format date and time
            dt = datetime.fromtimestamp(completion_time)
            date_int = int(dt.strftime('%Y%m%d00'))  # YYYYMMDD00 format
            time_utc = dt.strftime('%d %b %Y %H:%M:%S')
            
            row = {
                'loc': 4,
                'model': model,
                'account': 'benchmark',
                'date': date_int,
                'microtime': f'{completion_time:.4f}',
                'dc_id': 4,
                'dlog_tag': random.randint(100000000, 999999999),
                'layer': 1,
                'hostname': hostname,
                'source': 'benchmark',
                'event': 'translate_speed_stats',
                'data': '[]',
                'uldata': '[]' if result.success else '',
                'debug': json.dumps(debug_data),
                'time': int(completion_time),
                'uid': random.randint(1000000000, 9999999999),
                'time_utc': time_utc
            }
            
            writer.writerow(row)
    
    print(f"\nResults saved to: {csv_path}")
    print(f"  Total rows: {len(results)}")
    print(f"  Successful: {sum(1 for r in results if r.success)}")
    print(f"  Failed: {sum(1 for r in results if not r.success)}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark LLM translation endpoint')
    
    # Common translation arguments (without concurrency - we have different default)
    # Includes --config option
    add_translate_args(parser, include_concurrency=False)
    
    # Benchmark-specific arguments
    parser.add_argument('--concurrency', type=int, default=60,
                        help='Number of concurrent requests (or max workers for QPS mode)')
    parser.add_argument('--chunk-length', type=int, default=300,
                        help='Approximate length of each text chunk')
    parser.add_argument('--target-lang', default='German (de)',
                        help='Target language for translation')
    parser.add_argument('--max-chunks', type=int,
                        help='Maximum number of chunks to process (for testing)')
    parser.add_argument('--single-query', action='store_true',
                        help='Translate entire file as single query instead of chunks')
    parser.add_argument('--stats-interval', type=int, default=10,
                        help='Print interim stats every N requests')
    parser.add_argument('--log-file', type=str,
                        help='Parse queries from log file instead of War and Peace')
    parser.add_argument('--load-mode', default='fixed',
                        choices=['fixed', 'qps'],
                        help='Load generation mode: fixed (N active), qps (fixed rate)')
    parser.add_argument('--qps', type=float,
                        help='Target queries per second (required for --load-mode qps)')
    parser.add_argument('--query', type=str,
                        help='Single query text to translate repeatedly (use --max-chunks to limit)')
    parser.add_argument('--query-file', type=str,
                        help='Read query from file (use --max-chunks to limit repetitions)')
    parser.add_argument('--csv', type=str,
                        help='Save results to CSV file (same format as logging system)')

    args = parser.parse_args()

    if args.query and args.query_file:
        parser.error('Cannot specify both --query and --query-file')

    if args.load_mode == 'qps' and args.qps is None:
        parser.error('--qps is required when --load-mode is qps')

    # Prepare chunks and target languages
    target_langs = None

    if args.query or args.query_file:
        # Use query from command line or file
        if args.query_file:
            with open(args.query_file, 'r', encoding='utf-8') as f:
                query_text = f.read()
            print(f"Using query from file: {args.query_file} (length: {len(query_text)} chars)")
        else:
            query_text = args.query
            print(f"Using query from command line (length: {len(query_text)} chars)")

        repeat_count = args.max_chunks if args.max_chunks else 1000000  # Default to 1M if not specified
        print(f"Repeating query {repeat_count} times (or until stopped)")
        chunks = [query_text] * repeat_count
        print(f"  Query length: {len(query_text)} chars")
    elif args.log_file:
        print(f"Parsing queries from log file: {args.log_file}")
        queries = parse_log_file(args.log_file)
        print(f"Found {len(queries)} queries in log file")

        chunks = []
        target_langs = []
        for target_lang, texts in queries:
            for text in texts:
                chunks.append(text)
                target_langs.append(target_lang)

        if not chunks:
            print("No queries found in log file!")
            return

        print(f"\nPrepared {len(chunks)} texts for translation")
        chunk_lengths = [len(c) for c in chunks]
        print(f"  Average text length: {sum(chunk_lengths) / len(chunk_lengths):.0f} chars")
        print(f"  Total characters: {sum(chunk_lengths)}")
    else:
        text = download_war_and_peace()
        chunks = split_into_chunks(text, args.chunk_length, args.single_query)
        print(f"\nPrepared {len(chunks)} chunks for translation")
        if len(chunks) > 1:
            chunk_lengths = [len(c) for c in chunks]
            print(f"  Average chunk length: {sum(chunk_lengths) / len(chunk_lengths):.0f} chars")
            print(f"  Total characters: {sum(chunk_lengths)}")

    # Create config from file or args (config_from_args handles both)
    config = config_from_args(args)
    if args.config:
        print(f"Loaded config from: {args.config}")

    # Ensure target_langs is set for CSV output
    if target_langs is None:
        target_langs = [args.target_lang] * len(chunks)

    # Run benchmark
    results = run_benchmark(
        chunks=chunks,
        config=config,
        concurrency=args.concurrency,
        target_lang=args.target_lang,
        max_chunks=args.max_chunks,
        stats_interval=args.stats_interval,
        target_langs=target_langs,
        load_mode=args.load_mode,
        qps=args.qps
    )

    # Save to CSV if requested
    if args.csv:
        save_results_to_csv(
            results=results,
            all_chunks=chunks[:len(results)],  # Only chunks that were processed
            target_langs=target_langs[:len(results)],
            model=args.model,
            csv_path=args.csv
        )


if __name__ == "__main__":
    main()
