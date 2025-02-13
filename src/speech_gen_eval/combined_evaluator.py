"""
Copyright 2025 Balacoon

Combined evaluator executes multiple evaluators and returns a list of metrics
"""

import logging
import time

from speech_gen_eval.aesthetics import AestheticsEvaluator
from speech_gen_eval.ecapa2_secs import ECAPA2SECSEvaluator
from speech_gen_eval.evaluator import Evaluator
from speech_gen_eval.f0_accuracy import F0AccuracyEvaluator
from speech_gen_eval.f0_stats import F0StatsEvaluator
from speech_gen_eval.opensmile import OpenSmileEvaluator
from speech_gen_eval.utmosv2_quality import UTMOSv2QualityEvaluator
from speech_gen_eval.whisperv3_intelligibility import WhisperV3IntelligibilityEvaluator


class CombinedEvaluator(Evaluator):
    """
    Evaluator that combines multiple evaluators
    """

    def __init__(self, system_type, *args, **kwargs):
        """
        Initialize the evaluator
        Args:
            system_type (str): The type of system to evaluate
        """
        # default evaluators
        self._evaluators = [
            WhisperV3IntelligibilityEvaluator(*args, **kwargs),
            UTMOSv2QualityEvaluator(*args, **kwargs),
            AestheticsEvaluator(*args, **kwargs),
        ]
        if system_type == "tts":
            self._evaluators.extend(
                [
                    F0StatsEvaluator(*args, **kwargs),
                ]
            )
        elif system_type == "zero-tts" or system_type == "zero-vc":
            self._evaluators.extend(
                [
                    ECAPA2SECSEvaluator(*args, **kwargs),
                ]
            )
        elif system_type == "vocoder":
            self._evaluators.extend(
                [
                    F0AccuracyEvaluator(*args, **kwargs),
                    OpenSmileEvaluator(*args, **kwargs),
                ]
            )
        else:
            raise RuntimeError(f"Unexpected system type {system_type}")

    def get_metric(self) -> list[tuple[str, float]]:
        """
        Get the metrics for the evaluators,
        runs each evaluator
        Returns:
            list[tuple[str, float]]: A list of tuples, where each tuple contains a metric name and a value
        """
        metrics = []
        for eval in self._evaluators:
            start = time.time()
            metrics.extend(eval.get_metric())
            logging.info(f"It took {time.time() - start} to run {eval.get_info()}")
        return metrics
