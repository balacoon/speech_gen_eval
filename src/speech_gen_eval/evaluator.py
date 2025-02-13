"""
Copyright 2025 Balacoon

Evaluator - abstract class that does some objective measurement for audio
"""


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
