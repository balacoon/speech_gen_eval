"""
Copyright 2025, Balacoon

This evaluator computes opensmile features: jitter and shimmer.
"""

from functools import partial
from multiprocessing import Pool

import numpy as np
import opensmile

from speech_gen_eval.audio_dir import get_audio_paths
from speech_gen_eval.evaluator import Evaluator


def _process_fold(
    paths: list[str], ignore_errors: bool = False
) -> list[tuple[float, float]]:
    """Process a fold of audio files and return jitter/shimmer values"""
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    results = []
    for path in paths:
        try:
            features = smile.process_file(path)
            jitter = float(features["jitterLocal_sma3nz_amean"].iloc[0])
            shimmer = float(features["shimmerLocaldB_sma3nz_amean"].iloc[0])
            results.append((jitter, shimmer))
        except Exception as e:
            if not ignore_errors:
                raise e
    return results


class OpenSmileEvaluator(Evaluator):
    """
    OpenSmileEvaluator measures jitter and shimmer.
    These are metrics of periodicity of,
    which correlate with the quality of the speech signal.
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
        return "Jitter and Shimmer evaluation"

    def get_metric(self):
        """
        Get the metrics computed on all audio files
        """
        audio_paths = get_audio_paths(self._audio_dir, self._ids)

        # Split paths into folds for parallel processing
        fold_size = len(audio_paths) // self._njobs
        folds = [
            audio_paths[i : i + fold_size]
            for i in range(0, len(audio_paths), fold_size)
        ]

        # Process folds in parallel
        with Pool(self._njobs) as pool:
            fold_results = pool.map(
                partial(_process_fold, ignore_errors=self._ignore_errors), folds
            )

        # Combine results from all folds
        all_results = []
        for fold in fold_results:
            all_results.extend(fold)

        # Calculate mean jitter and shimmer
        if len(all_results) > 0:
            jitters, shimmers = zip(*all_results)
            return [
                ("jitter", float(np.mean(jitters))),
                ("shimmer", float(np.mean(shimmers))),
            ]
        return [("jitter", 0.0), ("shimmer", 0.0)]
