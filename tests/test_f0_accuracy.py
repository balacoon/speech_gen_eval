"""
Copyright 2025 Balacoon

Test f0 accuracy evaluator
"""

import os

from speech_gen_eval.f0_accuracy import F0AccuracyEvaluator
from speech_gen_eval.ids import read_txt_and_mapping


def test_f0_stats():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(test_dir, "assets", "txt")
    wav_path = os.path.join(test_dir, "assets", "wav")

    ids, _ = read_txt_and_mapping(txt_path, wav_path, ignore_missing=False)
    assert len(ids) == 10
    evaluator = F0AccuracyEvaluator(
        ids,
        generated_audio=wav_path,
        original_audio=wav_path,
        ignore_errors=False,
    )
    metrics = evaluator.get_metric()
    assert len(metrics) == 3
    for name, val in metrics:
        assert name.startswith("f0_")
        if name.endswith("_errors"):
            assert val == 0
        else:
            assert val == 1.0
