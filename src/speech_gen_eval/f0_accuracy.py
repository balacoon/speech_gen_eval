"""
Copyright 2025, Balacoon

This evaluator compares F0 in generated and reference audio.
"""

from functools import partial
from multiprocessing import Pool

import librosa
import numpy as np
from scipy.stats import pearsonr

from speech_gen_eval import evaluator
from speech_gen_eval.audio_dir import get_audio_paths


def _process_single_file(
    generated_path: str, original_path: str, ignore_errors: bool = False
) -> dict[str, float]:
    """
    Process a single audio file and return the f0 accuracy metrics
    """
    try:
        # Load audio
        y, _ = librosa.load(generated_path, sr=16000)
        y_ref, _ = librosa.load(original_path, sr=16000)

        # Compute F0 for both signals
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500)
        f0_ref, _, _ = librosa.pyin(y_ref, fmin=50, fmax=500)

        # Remove NaN values and take log
        valid_idx = ~np.isnan(f0) & ~np.isnan(f0_ref)
        log_f0 = np.log(f0[valid_idx])
        log_f0_ref = np.log(f0_ref[valid_idx])

        # Compute f0 errors
        f0_diff = np.abs(log_f0 - log_f0_ref)
        fine_errors = np.sum((f0_diff > 0) & (f0_diff < 0.2)) / len(
            f0_diff
        )  # 0 < diff < 0.2
        gross_errors = np.sum(f0_diff >= 0.2) / len(
            f0_diff
        )  # >= 20% threshold for gross errors

        # Compute correlation
        correlation, _ = pearsonr(log_f0, log_f0_ref)

        return {
            "fine_errors": fine_errors,
            "gross_errors": gross_errors,
            "correlation": correlation,
            "count": len(f0_diff),
        }
    except Exception as e:
        if not ignore_errors:
            raise e
        return None


class F0AccuracyEvaluator(evaluator.Evaluator):
    """
    F0AccuracyEvaluator.
    Measures:
    - f0_fine_errors
    - f0_gross_errors
    - f0_correlation
    """

    _njobs = 8

    def __init__(
        self,
        ids: dict[str, str],
        generated_audio: str,
        original_audio: str,
        ignore_errors: bool = True,
        **kwargs,
    ):
        self._ids = ids
        self._generated = generated_audio
        self._original = original_audio
        if self._original is None:
            raise ValueError("original_audio is required for F0 accuracy evaluation")
        self._ignore_errors = ignore_errors

    def get_info(self):
        """
        Get the info for the evaluator
        Returns:
            str: A string containing the info for the evaluator
        """
        return "F0 accuracy evaluation"

    def get_metric(self):
        """
        Get the metrics computed on all audio files
        """
        # Get paired paths for generated and original audio
        gen_paths = get_audio_paths(self._generated, self._ids)
        orig_paths = get_audio_paths(self._original, self._ids)
        paired_paths = list(zip(gen_paths, orig_paths))

        # Create a process pool
        with Pool(self._njobs) as pool:
            # Process files in parallel
            process_func = partial(
                _process_single_file, ignore_errors=self._ignore_errors
            )
            results = pool.starmap(process_func, paired_paths)

        # Filter out None results (from errors) and combine stats
        results = [r for r in results if r is not None]

        # Compute weighted averages based on number of valid F0 points
        total_count = sum(r["count"] for r in results)
        fine_errors = sum(r["fine_errors"] * r["count"] for r in results) / total_count
        gross_errors = (
            sum(r["gross_errors"] * r["count"] for r in results) / total_count
        )
        correlation = sum(r["correlation"] * r["count"] for r in results) / total_count

        return [
            ("f0_fine_errors", fine_errors),
            ("f0_gross_errors", gross_errors),
            ("f0_correlation", correlation),
        ]
