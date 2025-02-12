"""
Copyright 2025 Balacoon

UTMOSv2 - evaluate the quality of a speech system
"""

import os

import numpy as np
import torch
import utmosv2

from speech_gen_eval.evaluator import Evaluator


class UTMOSv2QualityEvaluator(Evaluator):
    """
    UTMOSv2 quality evaluator
    """

    _gpu_batch_size = 8

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

        self._model = utmosv2.create_model(pretrained=True)

    def get_info(self):
        """
        Get the info for the evaluator
        Returns:
            str: A string containing the info for the evaluator
        """
        return "Quality evaluation with UTMOSv2"

    def get_metric(self):
        """
        Get the metric for the evaluator
        Returns:
            list[tuple[str, float]]: A list of tuples, where each tuple contains a metric name and a value
        """
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        batch_size = self._gpu_batch_size if device == "cuda:0" else 1
        results = self._model.predict(
            input_dir=self._audio_dir, device=device, batch_size=batch_size
        )
        mos_dict = {
            os.path.splitext(os.path.basename(x["file_path"]))[0]: x["predicted_mos"]
            for x in results
        }
        mos_lst = [mos_dict[name] for name, _ in self._ids]
        return [("utmosv2_mos", np.mean(mos_lst))]
