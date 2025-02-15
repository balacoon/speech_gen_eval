"""
Copyright 2025 Balacoon

Combined evaluator executes multiple evaluators and returns a list of metrics
"""

import logging
import time

from speech_gen_eval.aesthetics import AestheticsEvaluator
from speech_gen_eval.ecapa_secs import ECAPA2SECSEvaluator, ECAPASECSEvaluator
from speech_gen_eval.evaluator import Evaluator
from speech_gen_eval.f0_accuracy import F0AccuracyEvaluator
from speech_gen_eval.f0_stats import F0StatsEvaluator
from speech_gen_eval.opensmile import OpenSmileEvaluator
from speech_gen_eval.utmos_quality import UTMOSQualityEvaluator
from speech_gen_eval.utmosv2_quality import UTMOSv2QualityEvaluator
from speech_gen_eval.whisperv3_intelligibility import WhisperV3IntelligibilityEvaluator

name2evaluator = {
    "utmos": UTMOSQualityEvaluator,
    "utmosv2": UTMOSv2QualityEvaluator,
    "cer": WhisperV3IntelligibilityEvaluator,
    "aesthetics": AestheticsEvaluator,
    "ecapa_secs": ECAPASECSEvaluator,
    "ecapa2_secs": ECAPA2SECSEvaluator,
    "f0accuracy": F0AccuracyEvaluator,
    "f0stats": F0StatsEvaluator,
    "jitter": OpenSmileEvaluator,
}
evaluator_names = sorted(list(name2evaluator.keys()))
type2names = {
    "tts": ["cer", "utmos", "aesthetics", "f0stats"],
    "zero-tts": ["cer", "utmos", "aesthetics", "ecapa_secs", "f0stats"],
    "zero-vc": ["cer", "utmos", "aesthetics", "ecapa_secs"],
    "vocoder": ["cer", "utmos", "aesthetics", "ecapa_secs", "f0accuracy", "jitter"],
}


class CombinedEvaluator(Evaluator):
    """
    Evaluator that combines multiple evaluators
    """

    def __init__(self, eval_names: list[str], *args, **kwargs):
        """
        Initialize the evaluator
        Args:
            system_type (str): The type of system to evaluate
        """
        self._evaluators = [
            name2evaluator[name](*args, **kwargs) for name in eval_names
        ]

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
