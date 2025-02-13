"""
Copyright 2025 Balacoon

Evaluate similarity between reference and generated audio using ECAPA2
"""

import numpy as np
import torch
import torchaudio
import tqdm
from huggingface_hub import hf_hub_download

from speech_gen_eval import evaluator
from speech_gen_eval.audio_dir import get_audio_path


class ECAPA2SECSEvaluator(evaluator.Evaluator):
    """
    ECAPA2 SECS evaluator
    """

    _model_id = "Jenthe/ECAPA2"

    def __init__(
        self,
        ids: dict[str, str],
        generated_audio: str,
        original_audio: str,
        mapping: dict[str, str] | None = None,
        ignore_errors: bool = True,
        **kwargs,
    ):
        self._ids = ids
        if mapping is None:
            # compare to itself if mapping is not provided.
            mapping = {x: x for x, _ in self._ids}
        self._mapping = mapping
        self._generated = generated_audio
        self._original = original_audio
        self._ignore_errors = ignore_errors

        model_file = hf_hub_download(
            repo_id="Jenthe/ECAPA2", filename="ecapa2.pt", cache_dir=None
        )
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._model = torch.jit.load(model_file, map_location=self._device)
        self._model.half()

    def get_info(self):
        """
        Get the info for the evaluator
        Returns:
            str: A string containing the info for the evaluator
        """
        return "Similarity evaluation with ECAPA2"

    def _extract_embedding(self, path: str) -> np.ndarray:
        arr, _ = torchaudio.load(path)
        arr = arr.to(torch.device(self._device))
        emb = self._model(arr)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().detach()

    def get_metric(self):
        """
        Get the metric for the evaluator
        Returns:
            list[tuple[str, float]]: A list of tuples, where each tuple contains a metric name and a value
        """

        # first extract the embeddings for reference audio
        ref_embeddings: dict[str, torch.Tensor] = {}
        for name, _ in tqdm.tqdm(self._ids):
            ref_name = self._mapping[name]
            if ref_name in ref_embeddings:
                continue
            ref_path = get_audio_path(self._original, ref_name)
            ref_embeddings[ref_name] = self._extract_embedding(ref_path)

        # now extract embeddings for generated audio and compare to reference
        secs_lst = []
        for name, _ in tqdm.tqdm(self._ids):
            ref_name = self._mapping[name]
            path = get_audio_path(self._generated, name)
            gen_emb = self._extract_embedding(path)
            ref_emb = ref_embeddings[ref_name]
            secs = torch.nn.functional.cosine_similarity(ref_emb, gen_emb).item()
            secs_lst.append(secs)

        return [("ecapa2_secs", np.mean(secs_lst))]
