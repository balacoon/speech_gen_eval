"""
Copyright 2025 Balacoon

Evaluate similarity between reference and generated audio using ECAPA/ECAPA2/ReDimNet
"""

import logging

import numpy as np
import soundfile as sf
import torch
import torchaudio
import tqdm
from huggingface_hub import hf_hub_download

from speech_gen_eval import evaluator
from speech_gen_eval.audio_dir import get_audio_path


class ECAPASECSEvaluator(evaluator.Evaluator):
    """
    ECAPA SECS evaluator
    """

    _model_name = "ecapa"

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
        if self._original is None:
            raise ValueError(
                f"original_audio is required for {self._model_name} SECS evaluation"
            )
        self._ignore_errors = ignore_errors
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        if self._device == "cpu":
            logging.warning("ECAPA model is not available on CPU")
            return None
        model_file = hf_hub_download(
            repo_id="balacoon/ecapa", filename="ecapa.jit", cache_dir=None
        )
        model = torch.jit.load(model_file).to(torch.device(self._device))
        return model

    def get_info(self):
        return f"Similarity evaluation with {self._model_name}"

    def _extract_embedding(self, model, path: str) -> np.ndarray:
        wav, sr = sf.read(path, dtype="int16")
        assert sr == 16000
        # run inference
        x = torch.tensor(wav).unsqueeze(0).cuda()
        x_len = torch.tensor([x.shape[1]], device=x.device)
        emb = model(x, x_len)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1).cpu().detach()
        return emb

    def get_metric(self):
        """
        Get the metric for the evaluator
        Returns:
            list[tuple[str, float]]: A list of tuples, where each tuple contains a metric name and a value
        """
        model = self._load_model()
        if model is None:
            logging.warning("ECAPA model is not available, SECS is not measured")
            return []
        # first extract the embeddings for reference audio
        ref_embeddings: dict[str, torch.Tensor] = {}
        for name, _ in tqdm.tqdm(self._ids):
            ref_name = self._mapping[name]
            if ref_name in ref_embeddings:
                continue
            ref_path = get_audio_path(self._original, ref_name)
            try:
                ref_embeddings[ref_name] = self._extract_embedding(model, ref_path)
            except Exception as e:
                logging.error(
                    f"Error exgracting reference spkr embedding for {ref_name}: {ref_path}"
                )
                if not self._ignore_errors:
                    raise e

        # now extract embeddings for generated audio and compare to reference
        secs_lst = []
        for name, _ in tqdm.tqdm(self._ids):
            ref_name = self._mapping[name]
            path = get_audio_path(self._generated, name)
            try:
                gen_emb = self._extract_embedding(model, path)
            except Exception as e:
                if not self._ignore_errors:
                    raise e
                logging.error(
                    f"Error exgracting generated spkr embedding for {name}: {path}"
                )
                continue
            ref_emb = ref_embeddings.get(ref_name, None)
            if ref_emb is None:
                msg = f"Reference spkr embedding for {ref_name} not found"
                if not self._ignore_errors:
                    raise ValueError(msg)
                logging.error(msg)
                continue
            secs = torch.nn.functional.cosine_similarity(ref_emb, gen_emb).item()
            secs_lst.append(secs)

        return [(f"{self._model_name}_secs", float(np.mean(secs_lst)))]


class ECAPA2SECSEvaluator(ECAPASECSEvaluator):
    """
    Same as parent class but uses newer speaker embedding model
    """

    _model_name = "ecapa2"

    def _load_model(self):
        model_file = hf_hub_download(
            repo_id="Jenthe/ECAPA2", filename="ecapa2.pt", cache_dir=None
        )
        model = torch.jit.load(model_file, map_location=self._device)
        if self._device == "cuda:0":
            model.half()
        return model

    def _extract_embedding(self, model, path: str) -> np.ndarray:
        arr, _ = torchaudio.load(path)
        arr = arr.to(torch.device(self._device))
        emb = model(arr)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().detach()


class ReDimNetSECSEvaluator(ECAPASECSEvaluator):
    """
    Same as parent class but uses ReDimNet
    """

    _model_name = "redimnet"

    def _load_model(self):
        model = torch.hub.load(
            "IDRnD/ReDimNet",
            "ReDimNet",
            model_name="b6",
            train_type="ft_lm",
            dataset="vox2",
        )
        if self._device == "cuda:0":
            model.cuda()
        model.eval()
        return model

    def _extract_embedding(self, model, path: str) -> np.ndarray:
        wav, sr = sf.read(path, dtype="float32")
        assert sr == 16000
        arr = torch.tensor(wav).unsqueeze(0)
        arr = arr.to(torch.device(self._device))
        emb = model(arr)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().detach()
