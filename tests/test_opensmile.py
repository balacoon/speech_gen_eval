"""
Copyright 2025 Balacoon

Test opensmile evaluator
"""

import os

from speech_gen_eval.opensmile import OpenSmileEvaluator
from speech_gen_eval.ids import read_txt_and_mapping


def test_opensmile():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(test_dir, "assets", "txt")
    wav_path = os.path.join(test_dir, "assets", "wav")

    ids, _ = read_txt_and_mapping(txt_path, wav_path, ignore_missing=False)
    assert len(ids) == 10
    evaluator = OpenSmileEvaluator(
        ids,
        wav_path,
        ignore_errors=False,
    )
    metrics = evaluator.get_metric()
    assert len(metrics) == 2
    for name, val in metrics:
        assert val > 0
        assert name in ["jitter", "shimmer"]
