"""
Copyright 2025 Balacoon

Main - entry point for speech generation evaluation
"""

import argparse
import logging

from speech_gen_eval.combined_evaluator import evaluator_names
from speech_gen_eval.evaluation import speech_gen_eval


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
    args = parse_args()
    speech_gen_eval(
        txt_path=args.txt,
        generated_audio=args.generated_audio,
        eval_type=args.type,
        original_audio=args.original_audio,
        mapping_path=args.mapping,
        evaluators=args.evaluators,
        ignore_missing=args.ignore_missing,
        out_path=args.out,
    )
