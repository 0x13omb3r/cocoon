#!/usr/bin/env python3
"""
Translation Quality Evaluation Script

Downloads WMT24++ benchmark data and evaluates translation quality using BLEU and chrF metrics.
Uses translate.py for translation.

Dataset: https://huggingface.co/datasets/google/wmt24pp

Usage:
    pip install sacrebleu datasets
    python quality_eval.py --endpoint http://127.0.0.1:8000 --pairs en-ru,en-zh
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

# Quality metrics
try:
    import sacrebleu
    from sacrebleu.metrics import BLEU, CHRF
except ImportError:
    print("Please install sacrebleu: pip install sacrebleu")
    sys.exit(1)

# COMET - neural metric (optional but recommended)
COMET_MODEL = None  # Global cache for COMET model
try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    print("Note: COMET not available. Install with: pip install unbabel-comet")
    print("      COMET gives better semantic evaluation than BLEU.\n")


def get_comet_model():
    """Load COMET model once and cache it."""
    global COMET_MODEL
    if COMET_MODEL is None and COMET_AVAILABLE:
        print("Loading COMET model (one-time)...")
        import warnings
        import logging
        # Suppress noisy warnings
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        logging.getLogger("torch").setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_path = download_model("Unbabel/wmt22-comet-da")
            COMET_MODEL = load_from_checkpoint(model_path)
        print("COMET model loaded.\n")
    return COMET_MODEL


def has_gpu() -> bool:
    """Check if GPU is available for COMET."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# Dataset download
try:
    from datasets import load_dataset
except ImportError:
    print("Please install datasets: pip install datasets")
    sys.exit(1)

from translate import translate, TranslateConfig, add_translate_args, config_from_args


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


@dataclass
class EvalResult:
    """Evaluation results for a language pair."""
    src_lang: str
    tgt_lang: str
    bleu: float
    chrf: float
    comet: Optional[float]
    num_samples: int
    num_errors: int
    avg_duration: float
    samples: List[TranslationSample] = field(default_factory=list)


def load_test_data(src_lang: str, tgt_lang: str, num_samples: int = 100) -> List[TranslationSample]:
    """
    Load test data for any language pair (with caching).
    
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
    
    # Check cache first
    if cache_key in _DATASET_CACHE:
        samples = _DATASET_CACHE[cache_key]
        print(f"Loading data: {src_lang}->{tgt_lang} (cached, {len(samples)} samples)")
        return [TranslationSample(source=s.source, reference=s.reference) for s in samples]
    
    # Use WMT24++ for en->xx (better quality post-edited references)
    if src_lang == "en" and tgt_lang in WMT_LANG_MAP:
        samples = _load_wmt_data(src_lang, tgt_lang, num_samples)
    else:
        # Use FLORES for xx->en and xx->yy
        samples = _load_flores_data(src_lang, tgt_lang, num_samples)
    
    # Cache for reuse
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


def translate_sample(
    sample: TranslationSample,
    tgt_lang: str,
    config: TranslateConfig,
    idx: int,
    total: int
) -> TranslationSample:
    """Translate a single sample."""
    target_lang_name = LANG_NAMES.get(tgt_lang, f"{tgt_lang}")
    
    start_time = time.time()
    try:
        result = translate(sample.source, target_lang=target_lang_name, config=config)
        sample.hypothesis = result.translation
        sample.duration = time.time() - start_time
        print(f"[{idx+1}/{total}] ✓ {sample.duration:.2f}s | {len(sample.source)} chars")
    except Exception as e:
        sample.error = str(e)
        sample.duration = time.time() - start_time
        print(f"[{idx+1}/{total}] ✗ {sample.duration:.2f}s | Error: {e}")
    
    return sample


def evaluate_pair(
    src_lang: str,
    tgt_lang: str,
    config: TranslateConfig,
    num_samples: int = 100,
    concurrency: int = 1,
    verbose: bool = False
) -> EvalResult:
    """
    Evaluate translation quality for a language pair.
    
    Args:
        src_lang: Source language code
        tgt_lang: Target language code
        config: TranslateConfig for translation (includes prompt_format)
        num_samples: Number of samples to evaluate
        concurrency: Number of concurrent requests
        verbose: Print sample details
    
    Returns:
        EvalResult with metrics
    """
    print(f"\n{'='*70}")
    print(f"Evaluating: {src_lang} -> {tgt_lang}")
    print(f"{'='*70}")
    
    # Load data
    samples = load_test_data(src_lang, tgt_lang, num_samples)
    
    if not samples:
        print(f"No samples loaded for {src_lang} -> {tgt_lang}")
        return EvalResult(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            bleu=0.0,
            chrf=0.0,
            comet=None,
            num_samples=0,
            num_errors=0,
            avg_duration=0.0
        )
    
    # Translate samples
    print(f"\nTranslating {len(samples)} samples...")
    start_time = time.time()
    
    if concurrency > 1:
        # Parallel translation
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    translate_sample, sample, tgt_lang, config, i, len(samples)
                ): i for i, sample in enumerate(samples)
            }
            for future in as_completed(futures):
                future.result()  # Raises exception if translation failed
    else:
        # Sequential translation
        for i, sample in enumerate(samples):
            translate_sample(sample, tgt_lang, config, i, len(samples))
    
    total_time = time.time() - start_time
    
    # Filter successful translations
    successful = [s for s in samples if s.hypothesis is not None]
    errors = [s for s in samples if s.error is not None]
    
    print(f"\nTranslation complete: {len(successful)} successful, {len(errors)} errors")
    print(f"Total time: {total_time:.2f}s ({total_time/len(samples):.2f}s per sample)")
    
    if not successful:
        return EvalResult(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            bleu=0.0,
            chrf=0.0,
            comet=None,
            num_samples=len(samples),
            num_errors=len(errors),
            avg_duration=total_time / len(successful) if successful else 0,
            samples=samples
        )
    
    # Calculate metrics
    hypotheses = [s.hypothesis for s in successful]
    references = [[s.reference] for s in successful]  # BLEU expects list of lists
    sources = [s.source for s in successful]
    
    # BLEU score
    bleu = BLEU()
    bleu_score = bleu.corpus_score(hypotheses, references)
    
    # chrF score
    chrf = CHRF()
    chrf_score = chrf.corpus_score(hypotheses, references)
    
    # COMET score (neural metric - much better for semantic evaluation)
    comet_score = None
    comet_scores_per_sample = None
    comet_model = get_comet_model()
    if comet_model is not None:
        print("\nCalculating COMET scores...")
        try:
            comet_data = [
                {"src": src, "mt": hyp, "ref": ref}
                for src, hyp, ref in zip(sources, hypotheses, [r[0] for r in references])
            ]
            comet_output = comet_model.predict(comet_data, batch_size=8, gpus=0)
            comet_score = comet_output.system_score
            comet_scores_per_sample = comet_output.scores
            print(f"COMET score: {comet_score:.4f}")
        except Exception as e:
            print(f"COMET calculation failed: {e}")
    
    avg_duration = sum(s.duration for s in successful) / len(successful) if successful else 0
    
    print(f"\n{'─'*70}")
    print(f"Results for {src_lang} -> {tgt_lang}:")
    print(f"  BLEU:  {bleu_score.score:.2f}")
    print(f"  chrF:  {chrf_score.score:.2f}")
    if comet_score is not None:
        print(f"  COMET: {comet_score:.4f}")
    print(f"  Samples: {len(successful)}/{len(samples)} successful")
    print(f"  Avg duration: {avg_duration:.2f}s (successful only)")
    print(f"{'─'*70}")
    
    # Always show some example translations with sentence-level scores
    print("\nExample translations:")
    num_examples = 5 if verbose else 3
    sent_bleu_metric = BLEU(effective_order=True)  # Better for short sentences
    for i, s in enumerate(successful[:num_examples]):
        # Calculate sentence-level scores
        sent_bleu = sent_bleu_metric.sentence_score(s.hypothesis, [s.reference])
        sent_chrf = chrf.sentence_score(s.hypothesis, [s.reference])
        
        # Include COMET if available
        if comet_scores_per_sample:
            comet_sent = comet_scores_per_sample[i]
            print(f"\n  [{i+1}] BLEU: {sent_bleu.score:.1f} | chrF: {sent_chrf.score:.1f} | COMET: {comet_sent:.3f}")
        else:
            print(f"\n  [{i+1}] BLEU: {sent_bleu.score:.1f} | chrF: {sent_chrf.score:.1f}")
        print(f"      Source:     {s.source[:200]}{'...' if len(s.source) > 200 else ''}")
        print(f"      Reference:  {s.reference[:200]}{'...' if len(s.reference) > 200 else ''}")
        print(f"      Hypothesis: {s.hypothesis[:200]}{'...' if len(s.hypothesis) > 200 else ''}")
    
    return EvalResult(
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        bleu=bleu_score.score,
        chrf=chrf_score.score,
        comet=comet_score,
        num_samples=len(samples),
        num_errors=len(errors),
        avg_duration=avg_duration,
        samples=samples
    )


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
    verbose: bool = False
) -> List[EvalResult]:
    """
    Batch evaluation: load all data, translate all, evaluate all.
    Much faster for COMET (single batch instead of per-pair).
    """
    # Phase 1: Load all data
    print(f"\n{'='*70}")
    print("PHASE 1: Loading all test data")
    print(f"{'='*70}")
    
    all_pair_data: List[PairData] = []
    for src, tgt in pairs:
        try:
            samples = load_test_data(src, tgt, num_samples)
            all_pair_data.append(PairData(src_lang=src, tgt_lang=tgt, samples=samples))
        except Exception as e:
            print(f"  Error loading {src}->{tgt}: {e}")
    
    total_samples = sum(len(pd.samples) for pd in all_pair_data)
    print(f"\nTotal: {total_samples} samples across {len(all_pair_data)} language pairs")
    
    # Phase 2: Translate all (interleaved across pairs)
    print(f"\n{'='*70}")
    print("PHASE 2: Translating all samples (interleaved)")
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
    
    def translate_task(task_idx, task):
        pair_idx, sample_idx, sample, tgt_lang = task
        pd = all_pair_data[pair_idx]
        pair_label = f"{pd.src_lang}->{pd.tgt_lang}"
        
        target_lang_name = LANG_NAMES.get(tgt_lang, f"{tgt_lang}")
        t0 = time.time()
        try:
            result = translate(sample.source, target_lang=target_lang_name, config=config)
            sample.hypothesis = result.translation
            sample.duration = time.time() - t0
            print(f"[{task_idx+1}/{total_samples}] {pair_label} ✓ {sample.duration:.2f}s | {len(sample.source)} chars")
        except Exception as e:
            sample.error = str(e)
            sample.duration = time.time() - t0
            error_msg = str(e)
            print(f"[{task_idx+1}/{total_samples}] {pair_label} ✗ {sample.duration:.2f}s | Error: {error_msg}")
            # Print more details for JSON errors
            if "Expecting value" in error_msg or "JSONDecode" in error_msg:
                print(f"      Source text: {sample.source[:100]}...")
                # Try to get response details from exception context
                if hasattr(e, '__context__') and e.__context__:
                    print(f"      Context: {str(e.__context__)[:200]}")
    
    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(translate_task, i, task): i 
                for i, task in enumerate(tasks)
            }
            for future in as_completed(futures):
                future.result()
    else:
        for i, task in enumerate(tasks):
            translate_task(i, task)
    
    translation_time = time.time() - start_time
    print(f"\nTranslation complete: {translation_time:.1f}s total ({translation_time/total_samples:.2f}s per sample)")
    
    # Phase 3: Evaluate all (COMET in single batch!)
    print(f"\n{'='*70}")
    print("PHASE 3: Calculating metrics")
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
    comet_model = get_comet_model()
    if comet_model is not None and all_successful:
        print(f"\nCalculating COMET for {len(all_successful)} samples (single batch)...")
        try:
            comet_data = [
                {"src": s.source, "mt": s.hypothesis, "ref": s.reference}
                for s in all_successful
            ]
            gpus = 1 if has_gpu() else 0
            comet_output = comet_model.predict(comet_data, batch_size=32, gpus=gpus, progress_bar=True)
            comet_scores_all = comet_output.scores
            print(f"COMET calculation complete.")
        except Exception as e:
            print(f"COMET calculation failed: {e}")
    
    # Calculate per-pair metrics
    print("\nCalculating per-pair metrics...")
    results = []
    bleu_metric = BLEU()
    chrf_metric = CHRF()
    sent_bleu_metric = BLEU(effective_order=True)
    
    for pair_idx, pd in enumerate(all_pair_data):
        successful = [s for s in pd.samples if s.hypothesis is not None]
        errors = [s for s in pd.samples if s.error is not None]
        
        if not successful:
            results.append(EvalResult(
                src_lang=pd.src_lang, tgt_lang=pd.tgt_lang,
                bleu=0.0, chrf=0.0, comet=None,
                num_samples=len(pd.samples), num_errors=len(errors),
                avg_duration=0.0, samples=pd.samples
            ))
            continue
        
        hypotheses = [s.hypothesis for s in successful]
        references = [[s.reference] for s in successful]
        
        bleu_score = bleu_metric.corpus_score(hypotheses, references)
        chrf_score = chrf_metric.corpus_score(hypotheses, references)
        
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
        print(f"  BLEU:  {bleu_score.score:.2f}")
        print(f"  chrF:  {chrf_score.score:.2f}")
        if comet_score is not None:
            print(f"  COMET: {comet_score:.4f}")
        print(f"  Samples: {len(successful)}/{len(pd.samples)} successful")
        
        # Show examples
        num_examples = 5 if verbose else 3
        print(f"\nExamples:")
        for i, s in enumerate(successful[:num_examples]):
            sent_bleu = sent_bleu_metric.sentence_score(s.hypothesis, [s.reference])
            sent_chrf = chrf_metric.sentence_score(s.hypothesis, [s.reference])
            
            if comet_scores_pair and i < len(comet_scores_pair):
                print(f"\n  [{i+1}] BLEU: {sent_bleu.score:.1f} | chrF: {sent_chrf.score:.1f} | COMET: {comet_scores_pair[i]:.3f}")
            else:
                print(f"\n  [{i+1}] BLEU: {sent_bleu.score:.1f} | chrF: {sent_chrf.score:.1f}")
            print(f"      Source:     {s.source[:200]}{'...' if len(s.source) > 200 else ''}")
            print(f"      Reference:  {s.reference[:200]}{'...' if len(s.reference) > 200 else ''}")
            print(f"      Hypothesis: {s.hypothesis[:200]}{'...' if len(s.hypothesis) > 200 else ''}")
        
        results.append(EvalResult(
            src_lang=pd.src_lang, tgt_lang=pd.tgt_lang,
            bleu=bleu_score.score, chrf=chrf_score.score, comet=comet_score,
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


def get_top_pairs_from_csv(csv_path: str, top_n: int = 20) -> List[Tuple[str, str]]:
    """
    Get top language pairs from lang.csv file.
    Excludes same-language pairs (en-en, ru-ru, etc.)
    Supports any direction (en->xx, xx->en, xx->yy).
    """
    import csv
    
    # Languages supported by FLORES-200 (haoranxu/FLORES-200)
    FLORES_LANGS = {"en", "ru", "zh", "es", "tr", "pt", "ko", "id", "ar", "fr", 
                    "vi", "ja", "it", "fa", "de", "uk", "uz", "pl", "nl", "he",
                    "cs", "hu", "sk", "sr", "th", "hi", "bn", "my", "el", "ro",
                    "bg", "da", "fi", "no", "sv", "et", "lt", "lv", "sl", "hr"}
    
    pairs = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            to_lang = row['to_lang']
            from_lang = row['from_lang']
            
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
                
            pairs.append((from_lang, to_lang))
            
            if len(pairs) >= top_n:
                break
    
    return pairs


def save_results(results: List[EvalResult], output_path: str):
    """Save evaluation results to JSON."""
    data = []
    for r in results:
        data.append({
            "src_lang": r.src_lang,
            "tgt_lang": r.tgt_lang,
            "bleu": r.bleu,
            "chrf": r.chrf,
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
    has_comet = any(r.comet is not None for r in results)
    
    print(f"\n{'='*85}")
    print(title)
    print(f"{'='*85}")
    if has_comet:
        print(f"{'Pair':<12} {'BLEU':>8} {'chrF':>8} {'COMET':>8} {'Samples':>10} {'Errors':>8} {'Avg Time':>10}")
    else:
        print(f"{'Pair':<12} {'BLEU':>8} {'chrF':>8} {'Samples':>10} {'Errors':>8} {'Avg Time':>10}")
    print(f"{'-'*85}")
    
    # Sort by COMET if available, otherwise by BLEU
    sort_key = (lambda x: x.comet or 0) if has_comet else (lambda x: x.bleu)
    for r in sorted(results, key=sort_key, reverse=True):
        pair = f"{r.src_lang}->{r.tgt_lang}"
        if has_comet:
            comet_str = f"{r.comet:.4f}" if r.comet is not None else "N/A"
            print(f"{pair:<12} {r.bleu:>8.2f} {r.chrf:>8.2f} {comet_str:>8} {r.num_samples:>10} {r.num_errors:>8} {r.avg_duration:>9.2f}s")
        else:
            print(f"{pair:<12} {r.bleu:>8.2f} {r.chrf:>8.2f} {r.num_samples:>10} {r.num_errors:>8} {r.avg_duration:>9.2f}s")
    
    print(f"{'-'*85}")
    
    # Averages
    if results:
        avg_bleu = sum(r.bleu for r in results) / len(results)
        avg_chrf = sum(r.chrf for r in results) / len(results)
        total_samples = sum(r.num_samples for r in results)
        total_errors = sum(r.num_errors for r in results)
        avg_duration = sum(r.avg_duration for r in results) / len(results)
        
        if has_comet:
            comet_results = [r.comet for r in results if r.comet is not None]
            avg_comet = sum(comet_results) / len(comet_results) if comet_results else 0
            print(f"{'AVERAGE':<12} {avg_bleu:>8.2f} {avg_chrf:>8.2f} {avg_comet:>8.4f} {total_samples:>10} {total_errors:>8} {avg_duration:>9.2f}s")
        else:
            print(f"{'AVERAGE':<12} {avg_bleu:>8.2f} {avg_chrf:>8.2f} {total_samples:>10} {total_errors:>8} {avg_duration:>9.2f}s")
    
    print(f"{'='*85}\n")


def print_comparison(results_local: List[EvalResult], results_azure: List[EvalResult]):
    """Print side-by-side comparison of local vs Azure results."""
    has_comet = any(r.comet is not None for r in results_local + results_azure)
    
    print(f"\n{'='*110}")
    print("COMPARISON: Local vs Azure")
    print(f"{'='*110}")
    
    if has_comet:
        print(f"{'Pair':<10} {'LOCAL COMET':>11} {'AZURE COMET':>11} {'Δ COMET':>9} │ {'LOCAL chrF':>10} {'AZURE chrF':>10} │ {'LOCAL t':>8} {'AZURE t':>8} {'Δ t':>8}")
    else:
        print(f"{'Pair':<10} {'LOCAL chrF':>10} {'AZURE chrF':>10} {'Δ chrF':>9} │ {'LOCAL t':>8} {'AZURE t':>8} {'Δ t':>8}")
    print(f"{'-'*110}")
    
    # Build lookup by pair
    local_by_pair = {(r.src_lang, r.tgt_lang): r for r in results_local}
    azure_by_pair = {(r.src_lang, r.tgt_lang): r for r in results_azure}
    
    all_pairs = set(local_by_pair.keys()) | set(azure_by_pair.keys())
    
    for pair in sorted(all_pairs):
        local = local_by_pair.get(pair)
        azure = azure_by_pair.get(pair)
        pair_str = f"{pair[0]}->{pair[1]}"
        
        local_time = local.avg_duration if local else 0
        azure_time = azure.avg_duration if azure else 0
        delta_time = local_time - azure_time
        delta_time_str = f"{delta_time:+.2f}s"
        
        if has_comet:
            local_comet = local.comet if local and local.comet else 0
            azure_comet = azure.comet if azure and azure.comet else 0
            delta_comet = local_comet - azure_comet
            delta_comet_str = f"{delta_comet:+.4f}"
            local_chrf = local.chrf if local else 0
            azure_chrf = azure.chrf if azure else 0
            print(f"{pair_str:<10} {local_comet:>11.4f} {azure_comet:>11.4f} {delta_comet_str:>9} │ {local_chrf:>10.2f} {azure_chrf:>10.2f} │ {local_time:>7.2f}s {azure_time:>7.2f}s {delta_time_str:>8}")
        else:
            local_chrf = local.chrf if local else 0
            azure_chrf = azure.chrf if azure else 0
            delta_chrf = local_chrf - azure_chrf
            delta_chrf_str = f"{delta_chrf:+.2f}"
            print(f"{pair_str:<10} {local_chrf:>10.2f} {azure_chrf:>10.2f} {delta_chrf_str:>9} │ {local_time:>7.2f}s {azure_time:>7.2f}s {delta_time_str:>8}")
    
    print(f"{'-'*110}")
    
    # Averages
    if results_local and results_azure:
        avg_local_time = sum(r.avg_duration for r in results_local) / len(results_local)
        avg_azure_time = sum(r.avg_duration for r in results_azure) / len(results_azure)
        delta_avg_time = avg_local_time - avg_azure_time
        delta_avg_time_str = f"{delta_avg_time:+.2f}s"
        
        if has_comet:
            local_comets = [r.comet for r in results_local if r.comet]
            azure_comets = [r.comet for r in results_azure if r.comet]
            avg_local_comet = sum(local_comets) / len(local_comets) if local_comets else 0
            avg_azure_comet = sum(azure_comets) / len(azure_comets) if azure_comets else 0
            delta_avg_comet = avg_local_comet - avg_azure_comet
            avg_local_chrf = sum(r.chrf for r in results_local) / len(results_local)
            avg_azure_chrf = sum(r.chrf for r in results_azure) / len(results_azure)
            print(f"{'AVERAGE':<10} {avg_local_comet:>11.4f} {avg_azure_comet:>11.4f} {delta_avg_comet:>+9.4f} │ {avg_local_chrf:>10.2f} {avg_azure_chrf:>10.2f} │ {avg_local_time:>7.2f}s {avg_azure_time:>7.2f}s {delta_avg_time_str:>8}")
        else:
            avg_local_chrf = sum(r.chrf for r in results_local) / len(results_local)
            avg_azure_chrf = sum(r.chrf for r in results_azure) / len(results_azure)
            delta_avg_chrf = avg_local_chrf - avg_azure_chrf
            print(f"{'AVERAGE':<10} {avg_local_chrf:>10.2f} {avg_azure_chrf:>10.2f} {delta_avg_chrf:>+9.2f} │ {avg_local_time:>7.2f}s {avg_azure_time:>7.2f}s {delta_avg_time_str:>8}")
    
    print(f"{'='*110}")
    print("Note: Positive Δ means LOCAL is better (higher COMET/chrF, lower time is shown as negative Δ t)")
    print()


def run_evaluation(
    pairs: List[Tuple[str, str]],
    config: TranslateConfig,
    num_samples: int,
    concurrency: int,
    verbose: bool,
    label: str = ""
) -> List[EvalResult]:
    """Run batch evaluation on all pairs and return results."""
    if label:
        print(f"\n{'#'*85}")
        print(f"# {label}")
        print(f"{'#'*85}")
    
    print(f"\nConfiguration:")
    print(f"  Endpoint: {config.endpoint}" + (" (Azure)" if config.use_azure else ""))
    print(f"  Model: {config.model}")
    print(f"  Samples per pair: {num_samples}")
    print(f"  Concurrency: {concurrency}")
    print(f"  GPU available: {has_gpu()}")
    
    # Use batch evaluation (load all -> translate all -> evaluate all)
    return evaluate_batch(
        pairs=pairs,
        config=config,
        num_samples=num_samples,
        concurrency=concurrency,
        verbose=verbose
    )


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate translation quality using WMT24++ benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate specific language pairs
  python quality_eval.py --pairs en-ru,en-zh --num-samples 50

  # Compare local endpoint vs Azure
  python quality_eval.py --pairs en-ru --compare --num-samples 50

  # Evaluate with Azure endpoint only
  python quality_eval.py --pairs en-ru --azure --num-samples 20
        """
    )
    
    # Common translation arguments
    add_translate_args(parser, include_concurrency=True)
    
    # Quality eval specific arguments
    parser.add_argument('--pairs', type=str,
                        help='Language pairs to evaluate, comma-separated (e.g., en-ru,en-zh)')
    parser.add_argument('--from-csv', type=str,
                        help='Load top language pairs from lang.csv file')
    parser.add_argument('--top-pairs', type=int, default=10,
                        help='Number of top pairs to evaluate from CSV')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples per language pair')
    parser.add_argument('--compare', action='store_true',
                        help='Compare local endpoint vs Azure (runs both)')
    parser.add_argument('--output', type=str, default='quality_results.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    # Determine language pairs
    if args.pairs:
        pairs = parse_lang_pairs(args.pairs)
    elif args.from_csv:
        pairs = get_top_pairs_from_csv(args.from_csv, args.top_pairs)
        print(f"Loaded {len(pairs)} pairs from {args.from_csv}")
    else:
        # Default: load top pairs from lang.csv in script directory
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_csv = os.path.join(script_dir, "lang.csv")
        
        if os.path.exists(default_csv):
            pairs = get_top_pairs_from_csv(default_csv, args.top_pairs)
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
        print(f"  {src} -> {tgt}")
    
    if args.compare:
        # Compare mode: run both local and Azure
        print("\n" + "="*85)
        print("COMPARISON MODE: Running evaluation on both Local and Azure endpoints")
        print("="*85)
        
        # Local endpoint
        config_local = config_from_args(args)
        config_local.use_azure = False
        results_local = run_evaluation(
            pairs, config_local,
            args.num_samples, args.concurrency, args.verbose,
            label=f"LOCAL ENDPOINT: {args.endpoint}"
        )
        
        # Azure endpoint (always use roles format)
        config_azure = config_from_args(args)
        config_azure.use_azure = True
        config_azure.prompt_format = "roles"
        results_azure = run_evaluation(
            pairs, config_azure,
            args.num_samples, args.concurrency, args.verbose,
            label="AZURE ENDPOINT"
        )
        
        # Print individual summaries
        if results_local:
            print_summary(results_local, title="SUMMARY: LOCAL")
        if results_azure:
            print_summary(results_azure, title="SUMMARY: AZURE")
        
        # Print comparison
        if results_local and results_azure:
            print_comparison(results_local, results_azure)
        
        # Save combined results
        if results_local or results_azure:
            combined = {
                "local": [{"src_lang": r.src_lang, "tgt_lang": r.tgt_lang, "bleu": r.bleu, "chrf": r.chrf, "comet": r.comet} for r in results_local],
                "azure": [{"src_lang": r.src_lang, "tgt_lang": r.tgt_lang, "bleu": r.bleu, "chrf": r.chrf, "comet": r.comet} for r in results_azure],
            }
            with open(args.output, 'w') as f:
                json.dump(combined, f, indent=2)
            print(f"\nResults saved to {args.output}")
    
    else:
        # Single endpoint mode
        config = config_from_args(args)
        
        # Azure always uses roles format
        if config.use_azure:
            config.prompt_format = "roles"
        
        results = run_evaluation(
            pairs, config,
            args.num_samples, args.concurrency, args.verbose
        )
        
        # Print summary and save
        if results:
            print_summary(results)
            save_results(results, args.output)


if __name__ == "__main__":
    main()

