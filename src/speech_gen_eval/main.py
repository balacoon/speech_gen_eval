"""
Copyright 2025 Balacoon

Main - entry point for speech generation evaluation
"""

import argparse
import logging
from typing import Optional

from speech_gen_eval.ids import read_txt_and_mapping
from speech_gen_eval.evaluator import CombinedEvaluator
from speech_gen_eval.audio_dir import sort_ids_by_audio_size, convert_audio_dir


def parse_args():
    """
    Parse command line arguments
    Returns:
        argparse.Namespace: The parsed arguments
    """
    ap = argparse.ArgumentParser(description="Runs speech generation evaluation")
    ap.add_argument("--generated-audio", required=True, help="Directory with generated audio to eval")
    ap.add_argument("--original-audio", help="Original audio")
    ap.add_argument("--mapping", help="Maps audio ids to reference ids")
    ap.add_argument("--txt", required=True, help="Text file with ids and text to run eval on")
    ap.add_argument("--type", choices=["tts", "zero-tts", "zero-vc", "vocoder"], default="zero-tts", help="Type of system to evaluate")
    ap.add_argument("--ignore-missing", action="store_true", help="Ignore when some id is missing or failed to process")
    args = ap.parse_args()

    # Conditional argument checks
    if args.type in ["zero-tts", "zero-vc", "vocoder"] and not args.original_audio:
        ap.error("--original-audio is required when type is 'zero-tts', 'zero-vc', or 'vocoder'.")

    if args.type in ["zero-tts", "zero-vc"] and not args.mapping:
        ap.error("--mapping is required when type is 'zero-tts' or 'zero-vc'.")

    return args


def main():
    """
    Main function
    """
    logging.basicConfig(level=logging.INFO)
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
            evaluator = CombinedEvaluator(
                args.type,
                txt=txt,
                generated_audio=generated_16khz,
                mapping=mapping,
                original_audio=original_16khz,
                ignore_errors=args.ignore_missing,
            )
            metrics = evaluator.get_metric()
            for metric in metrics:
                logging.info(f"{metric[0]}: {metric[1]:.4f}")