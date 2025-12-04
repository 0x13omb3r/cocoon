#!/usr/bin/env python3
"""
Translation Quality Evaluation Script

Downloads WMT24++ benchmark data and evaluates translation quality using COMET metric.
Uses translate.py for translation.

Dataset: https://huggingface.co/datasets/google/wmt24pp

Usage:
    pip install unbabel-comet datasets rich
    python quality_eval.py --configs azure.conf local.conf --pairs en-ru,en-zh
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pandas as pd

# Rich for table printing
from rich.console import Console
from rich.table import Table

# COMET - neural metric
from comet import download_model, load_from_checkpoint
COMET_MODEL = None  # Global cache for COMET model
COMET_MODEL_NAME = None  # Track which model is loaded

# Available COMET models
COMET_MODELS = {
    "wmt22": "Unbabel/wmt22-comet-da",           # Default, fast
    "xcomet-xl": "Unbabel/XCOMET-XL",            # Better quality
    "xcomet-xxl": "Unbabel/XCOMET-XXL",          # Best quality, requires more VRAM
}


def get_comet_model(model_name: str = "wmt22"):
    """Load COMET model once and cache it."""
    global COMET_MODEL, COMET_MODEL_NAME
    
    # If different model requested, reload
    if COMET_MODEL is not None and COMET_MODEL_NAME != model_name:
        COMET_MODEL = None
    
    if COMET_MODEL is None:
        model_id = COMET_MODELS.get(model_name, model_name)  # Allow direct model ID too
        print(f"Loading COMET model: {model_id} (one-time)...")
        import warnings
        import logging
        # Suppress noisy warnings
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        logging.getLogger("torch").setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_path = download_model(model_id)
            COMET_MODEL = load_from_checkpoint(model_path)
        COMET_MODEL_NAME = model_name
        print(f"COMET model loaded: {model_name}\n")
    return COMET_MODEL


def has_gpu() -> bool:
    """Check if GPU is available for COMET."""
    import torch
    return torch.cuda.is_available()


# Dataset download
from datasets import load_dataset

from translate import translate, TranslateConfig, config_from_args

# Translation cache using DuckDB
import duckdb


def _stable_hash(s: str) -> str:
    """Deterministic hash (first 16 bytes of sha256)."""
    import hashlib
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:32]


class TranslationCache:
    """Cache for translation results using DuckDB.
    
    Uses thread-local connections for thread safety.
    """
    
    def __init__(self, cache_path: Optional[str] = "translation_cache.duckdb"):
        self.cache_path = cache_path
        self.enabled = cache_path is not None
        self._local = threading.local()
        self._lock = threading.Lock()
        if self.enabled:
            self._init_db()
    
    def _get_conn(self):
        """Get thread-local connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = duckdb.connect(self.cache_path)
        return self._local.conn
    
    def _init_db(self):
        """Initialize DuckDB table."""
        try:
            conn = self._get_conn()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS translations (
                    source_hash VARCHAR,
                    target_lang VARCHAR,
                    config_key VARCHAR,
                    prompt_format VARCHAR,
                    hypothesis VARCHAR,
                    duration DOUBLE,
                    timestamp DOUBLE,
                    PRIMARY KEY (source_hash, target_lang, config_key, prompt_format)
                )
            """)
            count = conn.execute("SELECT COUNT(*) FROM translations").fetchone()[0]
            if count > 0:
                configs = conn.execute("SELECT DISTINCT config_key FROM translations LIMIT 5").fetchall()
                configs_str = ', '.join(c[0] for c in configs)
                print(f"Loaded {count} cached translations from {self.cache_path}")
                print(f"  Configs in cache: {configs_str}")
        except Exception as e:
            print(f"Warning: Could not initialize cache: {e}")
            self.enabled = False
    
    def get(self, source: str, target_lang: str, config: TranslateConfig, debug: bool = False) -> Optional[Tuple[str, float]]:
        """Get cached translation. Returns (hypothesis, duration) or None."""
        if not self.enabled:
            return None
        
        source_hash = _stable_hash(source)
        config_key = config.cache_key()
        
        if debug:
            print(f"  [cache] lookup: hash={source_hash[:16]}..., lang={target_lang}, key={config_key}, fmt={config.prompt_format}")
        
        try:
            conn = self._get_conn()
            result = conn.execute("""
                SELECT hypothesis, duration FROM translations 
                WHERE source_hash = ? AND target_lang = ? AND config_key = ? AND prompt_format = ?
            """, [source_hash, target_lang, config_key, config.prompt_format]).fetchone()
            
            if debug:
                count = conn.execute("SELECT COUNT(*) FROM translations").fetchone()[0]
                print(f"  [cache] {'HIT' if result else 'MISS'}, total entries: {count}")
            
            if result:
                return (result[0], result[1])
        except Exception as e:
            if debug:
                print(f"  [cache] error: {e}")
        return None
    
    def put(self, source: str, target_lang: str, config: TranslateConfig, hypothesis: str, duration: float):
        """Store translation in cache."""
        if not self.enabled:
            return
        
        source_hash = _stable_hash(source)
        config_key = config.cache_key()
        
        try:
            with self._lock:
                conn = self._get_conn()
                # INSERT OR REPLACE
                conn.execute("""
                    INSERT OR REPLACE INTO translations 
                    (source_hash, target_lang, config_key, prompt_format, hypothesis, duration, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, [source_hash, target_lang, config_key, config.prompt_format, hypothesis, duration, time.time()])
        except Exception as e:
            print(f"Warning: Could not cache translation: {e}")
    
    def save(self):
        """Explicitly save cache."""
        try:
            conn = self._get_conn()
            conn.execute("CHECKPOINT")
        except:
            pass
    
    def stats(self) -> str:
        """Return cache statistics."""
        if not self.enabled:
            return "disabled"
        try:
            conn = self._get_conn()
            count = conn.execute("SELECT COUNT(*) FROM translations").fetchone()[0]
            if count == 0:
                return "empty"
            configs = conn.execute("SELECT DISTINCT config_key FROM translations LIMIT 10").fetchall()
            configs_str = ', '.join(c[0] for c in configs)
            return f"{count} entries, configs: {configs_str}"
        except:
            return "error"


# Global cache instance
_TRANSLATION_CACHE: Optional[TranslationCache] = None

def get_translation_cache(cache_path: str = "translation_cache.parquet", enabled: bool = True) -> TranslationCache:
    """Get or create global translation cache."""
    global _TRANSLATION_CACHE
    if _TRANSLATION_CACHE is None:
        if enabled:
            _TRANSLATION_CACHE = TranslationCache(cache_path)
        else:
            _TRANSLATION_CACHE = TranslationCache(None)  # Disabled cache
    return _TRANSLATION_CACHE


def init_cache(cache_path: str, enabled: bool = True):
    """Initialize the global cache with specific settings."""
    global _TRANSLATION_CACHE
    if enabled and cache_path:
        _TRANSLATION_CACHE = TranslationCache(cache_path)
    else:
        _TRANSLATION_CACHE = TranslationCache(None)


# WMT24++ language code mapping
# Maps simple codes to WMT24++ config format (e.g., "ru" -> "ru_RU")
WMT_LANG_MAP = {
    "ar": "ar_EG",  # or ar_SA
    "bg": "bg_BG",
    "bn": "bn_IN",
    "ca": "ca_ES",
    "cs": "cs_CZ",
    "da": "da_DK",
    "de": "de_DE",
    "el": "el_GR",
    "es": "es_MX",
    "et": "et_EE",
    "fa": "fa_IR",
    "fi": "fi_FI",
    "fr": "fr_FR",  # or fr_CA
    "he": "he_IL",
    "hi": "hi_IN",
    "hr": "hr_HR",
    "hu": "hu_HU",
    "id": "id_ID",
    "is": "is_IS",
    "it": "it_IT",
    "ja": "ja_JP",
    "ko": "ko_KR",
    "lt": "lt_LT",
    "lv": "lv_LV",
    "nl": "nl_NL",
    "no": "no_NO",
    "pl": "pl_PL",
    "pt": "pt_BR",  # or pt_PT
    "ro": "ro_RO",
    "ru": "ru_RU",
    "sk": "sk_SK",
    "sl": "sl_SI",
    "sr": "sr_RS",
    "sv": "sv_SE",
    "th": "th_TH",
    "tr": "tr_TR",
    "uk": "uk_UA",
    "ur": "ur_PK",
    "vi": "vi_VN",
    "zh": "zh_CN",  # or zh_TW
    "zh-CN": "zh_CN",
    "zh-TW": "zh_TW",
}

# Human-readable language names for prompts
LANG_NAMES = {
    "en": "English (en)",
    "ru": "Russian (ru)",
    "zh": "Chinese (zh)",
    "zh-CN": "Chinese Simplified (zh-CN)",
    "zh-TW": "Chinese Traditional (zh-TW)",
    "es": "Spanish (es)",
    "tr": "Turkish (tr)",
    "pt": "Portuguese (pt)",
    "ko": "Korean (ko)",
    "id": "Indonesian (id)",
    "ar": "Arabic (ar)",
    "fr": "French (fr)",
    "vi": "Vietnamese (vi)",
    "ja": "Japanese (ja)",
    "it": "Italian (it)",
    "fa": "Persian (fa)",
    "de": "German (de)",
    "uk": "Ukrainian (uk)",
    "uz": "Uzbek (uz)",
    "pl": "Polish (pl)",
    "nl": "Dutch (nl)",
    "he": "Hebrew (he)",
    "cs": "Czech (cs)",
    "hu": "Hungarian (hu)",
    "sk": "Slovak (sk)",
    "sr": "Serbian (sr)",
    "th": "Thai (th)",
    "hi": "Hindi (hi)",
    "bn": "Bengali (bn)",
    "my": "Burmese (my)",
}


@dataclass
class TranslationSample:
    """A single translation sample with source, reference, and hypothesis."""
    source: str
    reference: str
    hypothesis: Optional[str] = None
    error: Optional[str] = None
    duration: float = 0.0


# Dataset cache (must be after TranslationSample is defined)
_DATASET_CACHE: Dict[str, List[TranslationSample]] = {}
_DATASET_CACHE_LOCK = threading.Lock()


@dataclass
class EvalResult:
    """Evaluation results for a language pair."""
    src_lang: str
    tgt_lang: str
    comet: Optional[float]
    num_samples: int
    num_errors: int
    avg_duration: float
    samples: List[TranslationSample] = field(default_factory=list)


def load_test_data(src_lang: str, tgt_lang: str, num_samples: int = 100) -> List[TranslationSample]:
    """
    Load test data for any language pair (with thread-safe caching).
    
    - Uses WMT24++ for en->xx pairs (better quality references)
    - Uses FLORES-200 for xx->en and xx->yy pairs
    
    Args:
        src_lang: Source language code
        tgt_lang: Target language code
        num_samples: Number of samples to load
    
    Returns:
        List of TranslationSample objects
    """
    cache_key = f"{src_lang}-{tgt_lang}:{num_samples}"
    
    # Check cache first (with lock for thread safety)
    with _DATASET_CACHE_LOCK:
        if cache_key in _DATASET_CACHE:
            samples = _DATASET_CACHE[cache_key]
            print(f"Loading data: {src_lang}->{tgt_lang} (cached, {len(samples)} samples)")
            return [TranslationSample(source=s.source, reference=s.reference) for s in samples]
    
    # Load outside lock (slow operation)
    if src_lang == "en" and tgt_lang in WMT_LANG_MAP:
        samples = _load_wmt_data(src_lang, tgt_lang, num_samples)
    else:
        samples = _load_flores_data(src_lang, tgt_lang, num_samples)
    
    # Cache for reuse (with lock)
    with _DATASET_CACHE_LOCK:
        _DATASET_CACHE[cache_key] = samples
    
    return [TranslationSample(source=s.source, reference=s.reference) for s in samples]


def _load_wmt_data(src_lang: str, tgt_lang: str, num_samples: int) -> List[TranslationSample]:
    """Load from WMT24++ (en->xx only)."""
    wmt_tgt = WMT_LANG_MAP.get(tgt_lang, tgt_lang)
    config = f"en-{wmt_tgt}"
    
    print(f"Loading WMT24++: {config}")
    dataset = load_dataset("google/wmt24pp", config, split="train")
    
    # Filter out bad source samples and limit text length to 200 chars
    good_samples = [
        row for row in dataset 
        if not row.get("is_bad_source", False) and len(row["source"]) <= 200
    ]
    print(f"  Filtered: {len(dataset)} -> {len(good_samples)} samples")
    
    samples = []
    for row in good_samples[:num_samples]:
        samples.append(TranslationSample(source=row["source"], reference=row["target"]))
    
    print(f"  Loaded {len(samples)} samples")
    return samples


def _load_flores_data(src_lang: str, tgt_lang: str, num_samples: int) -> List[TranslationSample]:
    """Load from FLORES-200 (any pair via English pivot)."""
    print(f"Loading FLORES: {src_lang}->{tgt_lang}")
    
    # haoranxu/FLORES-200 has configs like 'en-ru', 'ru-en'
    # For non-English pairs, we need to pivot through English
    
    if src_lang == "en":
        config = f"en-{tgt_lang}"
        dataset = load_dataset("haoranxu/FLORES-200", config, split="test")
        samples = []
        for i, row in enumerate(dataset):
            if i >= num_samples:
                break
            text = row[config]
            if len(text["en"]) <= 200:
                samples.append(TranslationSample(source=text["en"], reference=text[tgt_lang]))
    elif tgt_lang == "en":
        config = f"{src_lang}-en"
        dataset = load_dataset("haoranxu/FLORES-200", config, split="test")
        samples = []
        for i, row in enumerate(dataset):
            if i >= num_samples:
                break
            text = row[config]
            if len(text[src_lang]) <= 200:
                samples.append(TranslationSample(source=text[src_lang], reference=text["en"]))
    else:
        # Non-English pair: pivot through English (same sentence IDs)
        print(f"  Using English pivot for {src_lang}->{tgt_lang}")
        config_src = f"{src_lang}-en"
        config_tgt = f"en-{tgt_lang}"
        
        dataset_src = load_dataset("haoranxu/FLORES-200", config_src, split="test")
        dataset_tgt = load_dataset("haoranxu/FLORES-200", config_tgt, split="test")
        
        samples = []
        for i in range(min(num_samples, len(dataset_src), len(dataset_tgt))):
            src_text = dataset_src[i][config_src][src_lang]
            tgt_text = dataset_tgt[i][config_tgt][tgt_lang]
            if len(src_text) <= 200:
                samples.append(TranslationSample(source=src_text, reference=tgt_text))
    
    print(f"  Loaded {len(samples)} samples")
    return samples


@dataclass
class PairData:
    """Data for a single language pair evaluation."""
    src_lang: str
    tgt_lang: str
    samples: List[TranslationSample]


def evaluate_batch(
    pairs: List[Tuple[str, str]],
    config: TranslateConfig,
    num_samples: int = 100,
    concurrency: int = 1,
    verbose: bool = False,
    comet_model: str = "wmt22",
    label: str = ""
) -> List[EvalResult]:
    """
    Batch evaluation: load all data, translate all, evaluate all.
    Much faster for COMET (single batch instead of per-pair).
    """
    model_tag = f"[{label}] " if label else ""
    
    # Phase 1: Load all data (parallel)
    print(f"\n{'='*70}")
    print(f"{model_tag}PHASE 1: Loading all test data")
    print(f"{'='*70}")
    
    def load_pair(pair):
        src, tgt = pair
        try:
            samples = load_test_data(src, tgt, num_samples)
            return PairData(src_lang=src, tgt_lang=tgt, samples=samples)
        except Exception as e:
            print(f"  Error loading {src}->{tgt}: {e}")
            return None
    
    all_pair_data: List[PairData] = []
    with ThreadPoolExecutor(max_workers=min(8, len(pairs))) as executor:
        results = list(executor.map(load_pair, pairs))
        all_pair_data = [r for r in results if r is not None]
    
    total_samples = sum(len(pd.samples) for pd in all_pair_data)
    print(f"\n{model_tag}Total: {total_samples} samples across {len(all_pair_data)} language pairs")
    
    # Phase 2: Translate all (interleaved across pairs)
    print(f"\n{'='*70}")
    print(f"{model_tag}PHASE 2: Translating all samples (interleaved)")
    print(f"{'='*70}")
    
    # Build interleaved task list: round-robin across pairs
    # [(pair_idx, sample_idx, sample, tgt_lang), ...]
    tasks = []
    max_samples = max(len(pd.samples) for pd in all_pair_data)
    for sample_idx in range(max_samples):
        for pair_idx, pd in enumerate(all_pair_data):
            if sample_idx < len(pd.samples):
                tasks.append((pair_idx, sample_idx, pd.samples[sample_idx], pd.tgt_lang))
    
    start_time = time.time()
    
    def translate_task(task_idx, task, cache: TranslationCache):
        pair_idx, sample_idx, sample, tgt_lang = task
        pd = all_pair_data[pair_idx]
        pair_label = f"{pd.src_lang}->{pd.tgt_lang}"
        
        target_lang_name = LANG_NAMES.get(tgt_lang, f"{tgt_lang}")
        
        # Check cache first (debug for first 3 lookups)
        debug_cache = (task_idx < 3)
        cached = cache.get(sample.source, target_lang_name, config, debug=debug_cache)
        if cached:
            sample.hypothesis, sample.duration = cached
            print(f"{model_tag}[{task_idx+1}/{total_samples}] {pair_label} ⚡ CACHED ({sample.duration:.2f}s) | {len(sample.source)} chars")
            return
        
        t0 = time.time()
        try:
            result = translate(sample.source, target_lang=target_lang_name, config=config)
            sample.hypothesis = result.translation
            sample.duration = time.time() - t0
            print(f"{model_tag}[{task_idx+1}/{total_samples}] {pair_label} ✓ {sample.duration:.2f}s | {len(sample.source)} chars")
            # Store in cache
            cache.put(sample.source, target_lang_name, config, sample.hypothesis, sample.duration)
        except Exception as e:
            sample.error = str(e)
            sample.duration = time.time() - t0
            error_msg = str(e)
            print(f"{model_tag}[{task_idx+1}/{total_samples}] {pair_label} ✗ {sample.duration:.2f}s | Error: {error_msg}")
            if "Expecting value" in error_msg or "JSONDecode" in error_msg:
                print(f"      Source text: {sample.source[:100]}...")
    
    cache = get_translation_cache()
    
    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(translate_task, i, task, cache): i 
                for i, task in enumerate(tasks)
            }
            for future in as_completed(futures):
                future.result()
    else:
        for i, task in enumerate(tasks):
            translate_task(i, task, cache)
    
    # Save cache after translations
    cache.save()
    
    translation_time = time.time() - start_time
    print(f"\n{model_tag}Translation complete: {translation_time:.1f}s total ({translation_time/total_samples:.2f}s per sample)")
    print(f"{model_tag}Cache: {cache.stats()}")
    
    # Phase 3: Evaluate all (COMET in single batch!)
    print(f"\n{'='*70}")
    print(f"{model_tag}PHASE 3: Calculating metrics")
    print(f"{'='*70}")
    
    # Prepare all data for COMET (single batch)
    all_successful = []
    pair_indices = []  # Track which pair each sample belongs to
    
    for pair_idx, pd in enumerate(all_pair_data):
        for s in pd.samples:
            if s.hypothesis is not None:
                all_successful.append(s)
                pair_indices.append(pair_idx)
    
    # Calculate COMET for ALL samples at once
    comet_scores_all = None
    comet_model_instance = get_comet_model(comet_model)
    if comet_model_instance is not None and all_successful:
        print(f"\nCalculating COMET ({comet_model}) for {len(all_successful)} samples...")
        try:
            comet_data = [
                {"src": s.source, "mt": s.hypothesis, "ref": s.reference}
                for s in all_successful
            ]
            gpus = 1 if has_gpu() else 0
            comet_output = comet_model_instance.predict(comet_data, batch_size=32, gpus=gpus, progress_bar=True)
            comet_scores_all = comet_output.scores
            print(f"COMET calculation complete.")
        except Exception as e:
            print(f"COMET calculation failed: {e}")
    
    # Calculate per-pair metrics
    print("\nCalculating per-pair metrics...")
    results = []
    
    for pair_idx, pd in enumerate(all_pair_data):
        successful = [s for s in pd.samples if s.hypothesis is not None]
        errors = [s for s in pd.samples if s.error is not None]
        
        if not successful:
            results.append(EvalResult(
                src_lang=pd.src_lang, tgt_lang=pd.tgt_lang,
                comet=None,
                num_samples=len(pd.samples), num_errors=len(errors),
                avg_duration=0.0, samples=pd.samples
            ))
            continue
        
        # Get COMET scores for this pair
        comet_score = None
        comet_scores_pair = None
        if comet_scores_all is not None:
            pair_mask = [i for i, pi in enumerate(pair_indices) if pi == pair_idx]
            comet_scores_pair = [comet_scores_all[i] for i in pair_mask]
            comet_score = sum(comet_scores_pair) / len(comet_scores_pair) if comet_scores_pair else None
        
        successful_samples = [s for s in pd.samples if s.hypothesis is not None]
        avg_duration = sum(s.duration for s in successful_samples) / len(successful_samples) if successful_samples else 0
        
        # Print results for this pair
        print(f"\n{'─'*70}")
        print(f"Results for {pd.src_lang} -> {pd.tgt_lang}:")
        if comet_score is not None:
            print(f"  COMET: {comet_score:.4f}")
        print(f"  Samples: {len(successful)}/{len(pd.samples)} successful")
        
        # Show examples
        num_examples = 5 if verbose else 3
        print(f"\nExamples:")
        for i, s in enumerate(successful[:num_examples]):
            if comet_scores_pair and i < len(comet_scores_pair):
                print(f"\n  [{i+1}] COMET: {comet_scores_pair[i]:.3f}")
            else:
                print(f"\n  [{i+1}]")
            print(f"      Source:     {s.source[:200]}{'...' if len(s.source) > 200 else ''}")
            print(f"      Reference:  {s.reference[:200]}{'...' if len(s.reference) > 200 else ''}")
            print(f"      Hypothesis: {s.hypothesis[:200]}{'...' if len(s.hypothesis) > 200 else ''}")
        
        results.append(EvalResult(
            src_lang=pd.src_lang, tgt_lang=pd.tgt_lang,
            comet=comet_score,
            num_samples=len(pd.samples), num_errors=len(errors),
            avg_duration=avg_duration, samples=pd.samples
        ))
    
    return results


def parse_lang_pairs(pairs_str: str) -> List[Tuple[str, str]]:
    """Parse language pairs from comma-separated string like 'en-ru,ru-en,en-zh'."""
    pairs = []
    for pair in pairs_str.split(","):
        pair = pair.strip()
        if "-" in pair:
            src, tgt = pair.split("-", 1)
            pairs.append((src.strip(), tgt.strip()))
        else:
            print(f"Warning: Invalid pair format '{pair}', expected 'src-tgt'")
    return pairs


def get_top_pairs_from_csv(csv_path: str, top_n: int = 20) -> Tuple[List[Tuple[str, str]], Dict[str, float]]:
    """
    Get top language pairs from lang.csv file with cumulative percentages.
    Excludes same-language pairs (en-en, ru-ru, etc.)
    Supports any direction (en->xx, xx->en, xx->yy).
    
    Returns:
        Tuple of (pairs list, cumulative percentage dict keyed by "src->tgt")
    """
    import csv
    import re
    
    # Languages supported by FLORES-200 (haoranxu/FLORES-200)
    FLORES_LANGS = {"en", "ru", "zh", "es", "tr", "pt", "ko", "id", "ar", "fr", 
                    "vi", "ja", "it", "fa", "de", "uk", "uz", "pl", "nl", "he",
                    "cs", "hu", "sk", "sr", "th", "hi", "bn", "my", "el", "ro",
                    "bg", "da", "fi", "no", "sv", "et", "lt", "lv", "sl", "hr"}
    
    pairs = []
    cumulative_pct = {}
    total_pct = 0.0
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            to_lang = row['to_lang']
            from_lang = row['from_lang']
            
            # Parse percentage from num_pct column like "11317957 (7.11%)"
            pct_match = re.search(r'\(([\d.]+)%\)', row.get('num_pct', ''))
            pct = float(pct_match.group(1)) if pct_match else 0.0
            
            # Skip same-language pairs
            if to_lang == from_lang:
                continue
            
            # Normalize zh-CN/zh-TW to zh
            if from_lang in ("zh-CN", "zh-TW"):
                from_lang = "zh"
            if to_lang in ("zh-CN", "zh-TW"):
                to_lang = "zh"
            
            # Skip pairs not in FLORES
            if from_lang not in FLORES_LANGS or to_lang not in FLORES_LANGS:
                continue
            
            total_pct += pct
            pair_key = f"{from_lang}->{to_lang}"
            pairs.append((from_lang, to_lang))
            cumulative_pct[pair_key] = total_pct
            
            if len(pairs) >= top_n:
                break
    
    return pairs, cumulative_pct


# Global to store cumulative percentages for display
_PAIR_CUMULATIVE_PCT: Dict[str, float] = {}


def save_results(results: List[EvalResult], output_path: str):
    """Save evaluation results to JSON."""
    data = []
    for r in results:
        data.append({
            "src_lang": r.src_lang,
            "tgt_lang": r.tgt_lang,
            "comet": r.comet,
            "num_samples": r.num_samples,
            "num_errors": r.num_errors,
            "avg_duration": r.avg_duration
        })
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


def print_summary(results: List[EvalResult], title: str = "SUMMARY"):
    """Print summary table of all results."""
    console = Console()
    
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("Pair", style="cyan")
    table.add_column("COMET", justify="right")
    table.add_column("Samples", justify="right")
    table.add_column("Errors", justify="right")
    table.add_column("Avg Time", justify="right")
    
    # Sort by COMET
    for r in sorted(results, key=lambda x: x.comet or 0, reverse=True):
        pair = f"{r.src_lang}->{r.tgt_lang}"
        comet_str = f"{r.comet:.4f}" if r.comet is not None else "N/A"
        table.add_row(pair, comet_str, str(r.num_samples), str(r.num_errors), f"{r.avg_duration:.2f}s")
    
    # Averages
    if results:
        total_samples = sum(r.num_samples for r in results)
        total_errors = sum(r.num_errors for r in results)
        avg_duration = sum(r.avg_duration for r in results) / len(results)
        comet_results = [r.comet for r in results if r.comet is not None]
        avg_comet = sum(comet_results) / len(comet_results) if comet_results else 0
        
        table.add_section()
        table.add_row("AVERAGE", f"{avg_comet:.4f}", str(total_samples), str(total_errors), f"{avg_duration:.2f}s", style="bold")
    
    console.print(table)


def _format_value_rich(val: float, vals: list, higher_is_better: bool) -> str:
    """Format value with diff from best, using rich markup for colors."""
    best = max(vals) if higher_is_better else min(vals)
    worst = min(vals) if higher_is_better else max(vals)
    
    if val == best:
        return f"[green]{val:.2f}[/green]"
    else:
        diff_pct = ((val - best) / best) * 100 if best != 0 else 0
        text = f"{val:.2f} ({diff_pct:+.1f}%)"
        if val == worst:
            return f"[red]{text}[/red]"
        return text


def print_comparison(all_results: Dict[str, List[EvalResult]]):
    """Print comparison table for multiple models using rich."""
    if len(all_results) < 2:
        return
    
    console = Console()
    names = list(all_results.keys())
    
    # Preserve original pair order from first model's results
    first_results = list(all_results.values())[0]
    pair_order = [f"{r.src_lang}->{r.tgt_lang}" for r in first_results]
    
    # Build dataframe
    rows = []
    for name, results in all_results.items():
        for r in results:
            rows.append({
                'pair': f"{r.src_lang}->{r.tgt_lang}",
                'model': name,
                'score': r.comet if r.comet is not None else 0,
                'time': r.avg_duration,
                'errors': r.num_errors
            })
    
    df = pd.DataFrame(rows)
    
    # Check for duplicates
    duplicates = df.groupby(['pair', 'model']).size()
    duplicates = duplicates[duplicates > 1]
    if len(duplicates) > 0:
        console.print("\n[yellow]Warning: Duplicate entries found (will be aggregated):[/yellow]")
        for (pair, model), count in duplicates.items():
            console.print(f"  {pair} / {model}: {count} entries")
    
    # Use pivot_table to handle potential duplicates
    score_table = df.pivot_table(index='pair', columns='model', values='score', aggfunc='mean')[names]
    time_table = df.pivot_table(index='pair', columns='model', values='time', aggfunc='mean')[names]
    error_table = df.pivot_table(index='pair', columns='model', values='errors', aggfunc='sum')[names]
    
    # Create table
    err_legend = ", ".join(f"e{i+1}={n}" for i, n in enumerate(names))
    table = Table(title=f"COMPARISON: {' vs '.join(names)}\nErrors: {err_legend}", 
                  show_header=True, header_style="bold")
    
    # Add columns - left align so base numbers line up
    table.add_column("Pair", style="cyan")
    for name in names:
        table.add_column(f"{name}\nCOMET", justify="left")
    for name in names:
        table.add_column(f"{name[:6]}t\nsec", justify="left")
    for i in range(len(names)):
        table.add_column(f"e{i+1}", justify="right", width=4)
    
    # Data rows (in original order)
    for pair in pair_order:
        if pair not in score_table.index:
            continue
        scores = list(score_table.loc[pair])
        times = list(time_table.loc[pair])
        errors = list(error_table.loc[pair].astype(int))
        
        # Show cumulative percentage if available
        cum_pct = _PAIR_CUMULATIVE_PCT.get(pair)
        pair_display = f"{pair}({cum_pct:.0f}%)" if cum_pct is not None else pair
        
        row_data = [pair_display]
        for s in scores:
            row_data.append(_format_value_rich(s, scores, True))
        for t in times:
            row_data.append(_format_value_rich(t, times, False))
        for e in errors:
            row_data.append(f"[red]{e}[/red]" if e > 0 else str(e))
        table.add_row(*row_data)
    
    # Totals
    table.add_section()
    avg_scores = list(score_table.mean())
    avg_times = list(time_table.mean())
    total_errors = list(error_table.sum().astype(int))
    
    avg_row = ["AVERAGE"]
    for s in avg_scores:
        avg_row.append(_format_value_rich(s, avg_scores, True))
    for t in avg_times:
        avg_row.append(_format_value_rich(t, avg_times, False))
    for e in total_errors:
        avg_row.append(f"[red]{e}[/red]" if e > 0 else str(e))
    table.add_row(*avg_row, style="bold")
    
    console.print(table)
    console.print("[green]Green[/green] = Best, [red]Red[/red] = Worst/Errors")


def run_evaluation(
    pairs: List[Tuple[str, str]],
    config: TranslateConfig,
    num_samples: int,
    concurrency: int,
    verbose: bool,
    label: str = "",
    comet_model: str = "wmt22"
) -> List[EvalResult]:
    """Run batch evaluation on all pairs and return results."""
    if label:
        print(f"\n{'#'*85}")
        print(f"# {label}")
        print(f"{'#'*85}")
    
    print(f"\nConfiguration:")
    print(f"  Endpoint: {config.endpoint}" + (" (Azure)" if config.use_azure else ""))
    print(f"  Model: {config.model}")
    print(f"  Cache key: {config.cache_key()}")
    print(f"  Samples per pair: {num_samples}")
    print(f"  Concurrency: {concurrency}")
    print(f"  GPU available: {has_gpu()}")
    
    # Use batch evaluation (load all -> translate all -> evaluate all)
    return evaluate_batch(
        pairs=pairs,
        config=config,
        label=label,
        num_samples=num_samples,
        concurrency=concurrency,
        verbose=verbose,
        comet_model=comet_model
    )


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate translation quality using WMT24++ benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single model
  python quality_eval.py azure.conf --pairs en-ru

  # Compare two models
  python quality_eval.py tencent-local.conf azure.conf --pairs en-ru

  # Compare multiple models
  python quality_eval.py vllm.conf sglang.conf azure.conf --num-samples 50

Config file format (INI):
  [model]
  endpoint = http://127.0.0.1:10000
  model = Qwen/Qwen3-8B
  prompt_format = roles
  description = vllm-qwen3-8b
  timeout = 40
  azure = false
        """
    )
    
    # Config files (positional, one or more)
    parser.add_argument('configs', nargs='+', metavar='CONFIG',
                        help='Config file(s) to evaluate (INI format)')
    
    # Evaluation options
    parser.add_argument('--pairs', type=str,
                        help='Language pairs to evaluate, comma-separated (e.g., en-ru,en-zh)')
    parser.add_argument('--from-csv', type=str,
                        help='Load top language pairs from lang.csv file')
    parser.add_argument('--top-pairs', type=int, default=10,
                        help='Number of top pairs to evaluate from CSV')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples per language pair')
    parser.add_argument('--concurrency', type=int, default=1,
                        help='Number of concurrent translation requests')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--output', type=str, default='quality_results.json',
                        help='Output file for results')
    parser.add_argument('--cache', type=str, default='translation_cache.duckdb',
                        help='Translation cache file (duckdb format)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable translation caching')
    parser.add_argument('--comet-model', type=str, default='wmt22',
                        choices=['wmt22', 'xcomet-xl', 'xcomet-xxl'],
                        help='COMET model: wmt22 (fast), xcomet-xl (better), xcomet-xxl (best, needs GPU)')
    
    args = parser.parse_args()
    
    # Initialize cache
    init_cache(args.cache, enabled=not args.no_cache)
    
    # Determine language pairs
    global _PAIR_CUMULATIVE_PCT
    _PAIR_CUMULATIVE_PCT = {}
    
    if args.pairs:
        pairs = parse_lang_pairs(args.pairs)
    elif args.from_csv:
        pairs, _PAIR_CUMULATIVE_PCT = get_top_pairs_from_csv(args.from_csv, args.top_pairs)
        print(f"Loaded {len(pairs)} pairs from {args.from_csv}")
    else:
        # Default: load top pairs from lang.csv in script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_csv = os.path.join(script_dir, "lang.csv")
        
        if os.path.exists(default_csv):
            pairs, _PAIR_CUMULATIVE_PCT = get_top_pairs_from_csv(default_csv, args.top_pairs)
            print(f"Loaded top {len(pairs)} pairs from lang.csv")
        else:
            # Fallback if lang.csv not found
            pairs = [
                ("en", "ru"), ("en", "zh"), ("ru", "en"), ("en", "es"),
                ("en", "tr"), ("zh", "en"), ("en", "pt"), ("en", "ko"),
            ][:args.top_pairs]
            print(f"lang.csv not found, using {len(pairs)} default pairs")
    
    if not pairs:
        print("No language pairs to evaluate!")
        return
    
    print(f"\nLanguage pairs to evaluate ({len(pairs)}):")
    for src, tgt in pairs:
        pair_key = f"{src}->{tgt}"
        cum_pct = _PAIR_CUMULATIVE_PCT.get(pair_key)
        if cum_pct is not None:
            print(f"  {src} -> {tgt} (cumul: {cum_pct:.1f}%)")
        else:
            print(f"  {src} -> {tgt}")
    
    # Load all configs (with CLI overrides like --verbose)
    configs = []
    for config_path in args.configs:
        config = config_from_args(args, config_path=config_path)
        configs.append(config)
    
    print(f"\nModels to evaluate ({len(configs)}):")
    for config in configs:
        print(f"  {config.cache_key()}")
    
    # Run evaluation for each config
    all_results = {}
    for config in configs:
        name = config.cache_key()
        results = run_evaluation(
            pairs, config,
            args.num_samples, args.concurrency, args.verbose,
            label=name,
            comet_model=args.comet_model
        )
        all_results[name] = results
        if results:
            print_summary(results, title=f"SUMMARY: {name}")
    
    # Print comparison if multiple configs
    if len(configs) >= 2:
        valid_results = {k: v for k, v in all_results.items() if v}
        if len(valid_results) >= 2:
            print_comparison(valid_results)
    
    # Save results
    if any(all_results.values()):
        combined = {
            name: [{"src_lang": r.src_lang, "tgt_lang": r.tgt_lang, "comet": r.comet, "errors": r.num_errors, "avg_time": r.avg_duration} for r in results]
            for name, results in all_results.items() if results
        }
        with open(args.output, 'w') as f:
            json.dump(combined, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

