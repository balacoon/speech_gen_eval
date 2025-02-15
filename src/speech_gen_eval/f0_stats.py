"""
Copyright 2025, Balacoon

This evaluator computes the statistics of the F0 and rms of a speech signal.
"""

from functools import partial
from multiprocessing import Pool

import librosa
import numpy as np

from speech_gen_eval import evaluator
from speech_gen_eval.audio_dir import get_audio_paths


def _process_single_file(path: str, ignore_errors: bool = False) -> dict[str, float]:
    """
    Process a single audio file and return the f0 and rms statistics
    """
    try:
        # Load audio
        y, _ = librosa.load(path, sr=16000)

        # Compute F0
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500)
        log_f0 = np.log(f0[~np.isnan(f0)])

        # Compute f0 stats
        f0_sum = np.sum(log_f0)
        f0_sq_sum = np.sum(log_f0**2)
        f0_count = len(log_f0)

        # Compute f0 deltas
        f0_delta = librosa.feature.delta(log_f0)
        f0_delta_sum = np.sum(f0_delta)
        f0_delta_sq_sum = np.sum(f0_delta**2)
        f0_delta_count = len(f0_delta)

        # Compute loudness
        loudness = librosa.feature.rms(y=y)[0]
        loudness_sum = np.sum(loudness)
        loudness_sq_sum = np.sum(loudness**2)
        loudness_count = len(loudness)

        return {
            "f0_sum": f0_sum,
            "f0_sq_sum": f0_sq_sum,
            "f0_count": f0_count,
            "f0_delta_sum": f0_delta_sum,
            "f0_delta_sq_sum": f0_delta_sq_sum,
            "f0_delta_count": f0_delta_count,
            "loudness_sum": loudness_sum,
            "loudness_sq_sum": loudness_sq_sum,
            "loudness_count": loudness_count,
        }
    except Exception as e:
        if not ignore_errors:
            raise e
        return None


class F0StatsEvaluator(evaluator.Evaluator):
    """
    F0StatsEvaluator.
    Measures:
    - f0_std: Standard deviation of the F0.
    - f0_delta_std: Standard deviation of the delta of the F0.
    - loudness_std: Standard deviation of the rms.
    """

    _njobs = 8

    def __init__(
        self,
        ids: dict[str, str],
        generated_audio: str,
        ignore_errors: bool = True,
        **kwargs,
    ):
        self._ids = ids
        self._audio_dir = generated_audio
        self._ignore_errors = ignore_errors

    def get_info(self):
        """
        Get the info for the evaluator
        Returns:
            str: A string containing the info for the evaluator
        """
        return "F0 and RMS statistics as expessivity evaluation"

    def get_metric(self):
        """
        Get the metrics computed on all audio files
        """
        audio_paths = get_audio_paths(self._audio_dir, self._ids)
        # Create a process pool
        with Pool(self._njobs) as pool:
            # Process files in parallel
            process_func = partial(
                _process_single_file, ignore_errors=self._ignore_errors
            )
            results = pool.map(process_func, audio_paths)

        # Filter out None results (from errors) and combine stats
        results = [r for r in results if r is not None]

        # Sum up all the accumulators
        f0_sum = sum(r["f0_sum"] for r in results)
        f0_sq_sum = sum(r["f0_sq_sum"] for r in results)
        f0_count = sum(r["f0_count"] for r in results)
        f0_delta_sum = sum(r["f0_delta_sum"] for r in results)
        f0_delta_sq_sum = sum(r["f0_delta_sq_sum"] for r in results)
        f0_delta_count = sum(r["f0_delta_count"] for r in results)
        loudness_sum = sum(r["loudness_sum"] for r in results)
        loudness_sq_sum = sum(r["loudness_sq_sum"] for r in results)
        loudness_count = sum(r["loudness_count"] for r in results)

        # Compute overall statistics
        f0_mean = f0_sum / f0_count
        f0_std = np.sqrt(f0_sq_sum / f0_count - f0_mean**2)

        f0_delta_mean = f0_delta_sum / f0_delta_count
        f0_delta_std = np.sqrt(f0_delta_sq_sum / f0_delta_count - f0_delta_mean**2)

        loudness_mean = loudness_sum / loudness_count
        loudness_std = np.sqrt(loudness_sq_sum / loudness_count - loudness_mean**2)

        return [
            ("log_f0_std", float(f0_std)),
            ("log_f0_delta_std", float(f0_delta_std)),
            ("loudness_std", float(loudness_std)),
        ]
