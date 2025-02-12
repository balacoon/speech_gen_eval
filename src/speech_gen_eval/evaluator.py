"""
Copyright 2025 Balacoon

Evaluator - abstract class that does some objective measurement for audio
"""

import logging
import time


class Evaluator:
    """
    Abstract class for evaluators
    """

    def get_metric(self) -> list[tuple[str, float]]:
        """
        Get the metric for the evaluator
        Returns:
            list[tuple[str, float]]: A list of tuples, where each tuple contains a metric name and a value
        """
        pass

    def get_info(self) -> str:
        """
        Get the info for the evaluator
        Returns:
            str: A string containing the info for the evaluator
        """
        pass


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
        self._evaluators = []
        if system_type == "tts":
            pass
        elif system_type == "zero-tts":
            pass
        elif system_type == "zero-vc":
            pass
        elif system_type == "vocoder":
            pass
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
