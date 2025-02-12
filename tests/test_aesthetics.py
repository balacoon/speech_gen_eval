"""
Copyright 2025 Balacoon

Test - test the Aesthetics evaluator
"""

import os

from speech_gen_eval.aesthetics import AestheticsEvaluator
from speech_gen_eval.ids import read_txt_and_mapping


def test_quality():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(test_dir, "assets", "txt")
    wav_path = os.path.join(test_dir, "assets", "wav")

    ids, _ = read_txt_and_mapping(txt_path, wav_path, ignore_missing=False)
    assert len(ids) == 10
    evaluator = AestheticsEvaluator(
        ids,
        wav_path,
        ignore_errors=False,
    )
    metrics = evaluator.get_metric()
    assert len(metrics) == 4
    for name, val in metrics:
        assert name.startswith("aesthetics_")
        assert val > 0
