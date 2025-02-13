"""
Copyright 2025 Balacoon

Aesthetics evaluator
https://ai.meta.com/research/publications/meta-audiobox-aesthetics-unified-automatic-quality-assessment-for-speech-music-and-sound/
"""

import json
import os

import numpy as np
import torch
import tqdm
from audiobox_aesthetics.cli import DEFAULT_CKPT_URL, download_file
from audiobox_aesthetics.infer import AesWavlmPredictorMultiOutput

from speech_gen_eval import evaluator
from speech_gen_eval.audio_dir import get_audio_paths


class AestheticsEvaluator(evaluator.Evaluator):
    """
    Aesthetics evaluator.
    Measures:
    - CE: Content Enjoyment
    - CU: Content Usefulness
    - PC: Production Complexity
    - PQ: Production Quality
    """

    _gpu_batch_size = 8
    _local_ckpt_path = os.path.expanduser(
        "~/.cache/audiobox_aesthetics/audiobox_aesthetics.pth"
    )

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

        if not os.path.isfile(self._local_ckpt_path):
            os.makedirs(os.path.dirname(self._local_ckpt_path), exist_ok=True)
            download_file(DEFAULT_CKPT_URL, self._local_ckpt_path)

        self._model = AesWavlmPredictorMultiOutput(self._local_ckpt_path)
        self._model.setup_model()

    def get_info(self):
        """
        Get the info for the evaluator
        Returns:
            str: A string containing the info for the evaluator
        """
        return "Aesthetic evaluation"

    def get_metric(self):
        """
        Get the metric
        """
        audio_paths = get_audio_paths(self._audio_dir, self._ids)
        metadata = [{"path": path} for path in audio_paths]
        outputs = []
        batch_size = self._gpu_batch_size if torch.cuda.is_available() else 1
        for ii in tqdm.tqdm(range(0, len(metadata), batch_size)):
            results_str = self._model.forward(metadata[ii : ii + batch_size])
            results = [json.loads(x) for x in results_str]
            outputs.extend(results)

        metrics = []
        for key, name in [
            ("CE", "enjoyment"),
            ("CU", "usefullness"),
            ("PC", "complexity"),
            ("PQ", "quality"),
        ]:
            values = [x[key] for x in outputs]
            metrics.append((f"aesthetics_{name}", np.mean(values)))
        return metrics
