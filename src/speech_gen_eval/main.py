"""
Copyright 2025 Balacoon

Main - entry point for speech generation evaluation
"""

import argparse
import logging
import warnings

import yaml

from speech_gen_eval.audio_dir import convert_audio_dir, sort_ids_by_audio_size
from speech_gen_eval.combined_evaluator import (
    CombinedEvaluator,
    evaluator_names,
    type2names,
)
from speech_gen_eval.ids import read_txt_and_mapping


def parse_args():
    """
    Parse command line arguments
    Returns:
        argparse.Namespace: The parsed arguments
    """
    ap = argparse.ArgumentParser(description="Runs speech generation evaluation")
    ap.add_argument(
        "--generated-audio",
        required=True,
        help="Directory with generated audio to eval",
    )
    ap.add_argument("--original-audio", help="Original audio")
    ap.add_argument("--mapping", help="Maps audio ids to reference ids")
    ap.add_argument(
        "--txt", required=True, help="Text file with ids and text to run eval on"
    )
    ap.add_argument(
        "--type",
        choices=["tts", "zero-tts", "zero-vc", "vocoder", "custom"],
        default="zero-tts",
        help="Type of system to evaluate",
    )
    ap.add_argument(
        "--evaluators",
        nargs="+",
        choices=evaluator_names,
        help="If running custom evaluation, specify the evaluators to run",
    )
    ap.add_argument(
        "--ignore-missing",
        action="store_true",
        help="Ignore when some id is missing or failed to process",
    )
    ap.add_argument("--out", help="Output file to save metrics")
    args = ap.parse_args()

    # Conditional argument checks
    if args.type in ["zero-tts", "zero-vc", "vocoder"] and not args.original_audio:
        ap.error(
            "--original-audio is required when type is 'zero-tts', 'zero-vc', or 'vocoder'."
        )

    if args.type in ["zero-tts", "zero-vc"] and not args.mapping:
        ap.error("--mapping is required when type is 'zero-tts' or 'zero-vc'.")

    if args.type != "custom" and args.evaluators:
        ap.error("--evaluators is only allowed when type is 'custom'.")

    if args.type == "custom" and not args.evaluators:
        ap.error("--evaluators is required when type is 'custom'.")

    return args


def main():
    """
    Main function
    """
    logging.basicConfig(level=logging.INFO)
    # supress warnings from torch and transformers
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")

    args = parse_args()
    txt, mapping = read_txt_and_mapping(
        args.txt,
        args.generated_audio,
        mapping_path=args.mapping,
        original_audio=args.original_audio,
        ignore_missing=args.ignore_missing,
    )
    txt = sort_ids_by_audio_size(args.generated_audio, txt)
    with convert_audio_dir(args.generated_audio, txt, 16000) as generated_16khz:
        with convert_audio_dir(args.original_audio, txt, 16000) as original_16khz:
            if args.type == "custom":
                evaluator_names = args.evaluators
            else:
                evaluator_names = type2names[args.type]
            evaluator = CombinedEvaluator(
                evaluator_names,
                ids=txt,
                generated_audio=generated_16khz,
                mapping=mapping,
                original_audio=original_16khz,
                ignore_errors=args.ignore_missing,
            )
            metrics = evaluator.get_metric()
            for metric in metrics:
                logging.info(f"{metric[0]}: {metric[1]:.4f}")

    if args.out:
        # Convert metrics to dict to save into yaml
        metrics_dict = dict(metrics)
        with open(args.out, "w") as f:
            yaml.dump(metrics_dict, f, default_flow_style=False)
