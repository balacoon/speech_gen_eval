"""
Copyright 2025 Balacoon

Intelligibility - evaluate the intelligibility of a speech system
"""

import jiwer
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from speech_gen_eval import evaluator
from speech_gen_eval.audio_dir import get_audio_paths


class WhisperV3IntelligibilityEvaluator(evaluator.Evaluator):
    """
    Intelligibility evaluator using Whisper V3
    """

    _model_id = "openai/whisper-large-v3-turbo"
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

        # uses whisper-large-v3-turbo
        # https://huggingface.co/openai/whisper-large-v3-turbo
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._batch_size = self._gpu_batch_size if self._device == "cuda:0" else 1
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self._model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(self._device)
        processor = AutoProcessor.from_pretrained(self._model_id)
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=self._device,
        )

    def get_info(self):
        """
        Get the info for the evaluator
        Returns:
            str: A string containing the info for the evaluator
        """
        return f"Intelligibility evaluation with {self._model_id}"

    def get_metric(self):
        """
        Get the metric for the evaluator
        Returns:
            list[tuple[str, float]]: A list of tuples, where each tuple contains a metric name and a value
        """
        audio_paths = get_audio_paths(self._audio_dir, self._ids)
        results = self._pipe(audio_paths, batch_size=self._batch_size)
        ref_txt_lst = [x[1] for x in self._ids]
        hyp_txt_lst = [x.get("text", "") for x in results]
        cer = jiwer.cer(ref_txt_lst, hyp_txt_lst)
        return [("whisperv3_cer", cer)]
