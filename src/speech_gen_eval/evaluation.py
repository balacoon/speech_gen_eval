"""
Copyright 2025 Balacoon

Main evaluation function
"""

import logging
import warnings

import yaml

# supress warnings from torch and transformers
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

from speech_gen_eval.audio_dir import convert_audio_dir, sort_ids_by_audio_size
from speech_gen_eval.combined_evaluator import (
    CombinedEvaluator,
    type2names,
)
from speech_gen_eval.ids import read_txt_and_mapping


def speech_gen_eval(
    txt_path: str,
    generated_audio: str,
    eval_type: str,
    original_audio: str | None = None,
    mapping_path: str | None = None,
    evaluators: list[str] | None = None,
    ignore_missing: bool = False,
    out_path: str | None = None,
    **kwargs,
) -> list[tuple[str, float]]:
    """
    Run speech generation evaluation
    Args:
        txt_path: Path to text file with ids and text
        generated_audio: Directory with generated audio to evaluate
        eval_type: Type of system to evaluate
        original_audio: Path to original audio directory
        mapping_path: Path to mapping file
        evaluators: List of evaluators for custom evaluation
        ignore_missing: Whether to ignore missing/failed files
        out_path: Output file to save metrics
        **kwargs: Additional fields to be saved to the output file
    Returns:
        List of (metric_name, value) tuples
    """
    txt, mapping = read_txt_and_mapping(
        txt_path,
        generated_audio,
        mapping_path=mapping_path,
        original_audio=original_audio,
        ignore_missing=ignore_missing,
    )
    txt = sort_ids_by_audio_size(generated_audio, txt)

    with convert_audio_dir(generated_audio, txt, sample_rate=16000) as generated_16khz:
        with convert_audio_dir(
            original_audio, txt, mapping=mapping, sample_rate=16000
        ) as original_16khz:
            if eval_type == "custom":
                eval_names = evaluators
            else:
                eval_names = type2names[eval_type]
            evaluator = CombinedEvaluator(
                eval_names,
                ids=txt,
                generated_audio=generated_16khz,
                mapping=mapping,
                original_audio=original_16khz,
                ignore_errors=ignore_missing,
            )
            metrics = evaluator.get_metric()
            for metric in metrics:
                logging.info(f"{metric[0]}: {metric[1]:.4f}")

    if out_path:
        output_dict = {"metrics": dict(metrics), **kwargs}
        with open(out_path, "w") as f:
            yaml.dump(output_dict, f, default_flow_style=False)

    return metrics
