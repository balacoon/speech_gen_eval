"""
Copyright 2025 Balacoon

UTMOS - evaluate the quality of a speech system
"""

import logging

import numpy as np
import soundfile as sf
import torch
import tqdm
from huggingface_hub import hf_hub_download

from speech_gen_eval import evaluator
from speech_gen_eval.audio_dir import get_audio_paths


class UTMOSQualityEvaluator(evaluator.Evaluator):
    """
    UTMOSv2 quality evaluator
    """

    _gpu_batch_size = 4

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

        if not torch.cuda.is_available():
            logging.warning("UTMOS is a GPU-only model")
        else:
            model_path = hf_hub_download(repo_id="balacoon/utmos", filename="utmos.jit")
            self._model = torch.jit.load(model_path)

    def get_info(self):
        """
        Get the info for the evaluator
        Returns:
            str: A string containing the info for the evaluator
        """
        return "Quality evaluation with UTMOS"

    def get_metric(self):
        """
        Get the metric for the evaluator
        Returns:
            list[tuple[str, float]]: A list of tuples, where each tuple contains a metric name and a value
        """
        if not torch.cuda.is_available():
            logging.warning("no GPU, UTMOS metric is not computed")
            return []

        audio_paths = get_audio_paths(self._audio_dir, self._ids)
        all_scores = []

        # Process in batches
        for i in tqdm.tqdm(range(0, len(audio_paths), self._gpu_batch_size)):
            batch_paths = audio_paths[i : i + self._gpu_batch_size]

            # Load audio files
            batch_audio = []
            for batch_path in batch_paths:
                audio, sr = sf.read(batch_path, dtype="int16")
                assert sr == 16000
                batch_audio.append(torch.from_numpy(audio))

            # Pad batch to max length
            x = torch.nn.utils.rnn.pad_sequence(batch_audio, batch_first=True)
            x = x.to("cuda")

            # Get predictions
            with torch.no_grad():
                scores = self._model(x)

            # Move back to CPU and collect results
            all_scores.extend(scores.detach().cpu().tolist())

        mean_score = float(np.mean(all_scores))
        return [("utmos_mos", mean_score)]
