"""
Copyright 2025 Balacoon

Test f0 stats evaluator
"""

import os

from speech_gen_eval.f0_stats import F0StatsEvaluator
from speech_gen_eval.ids import read_txt_and_mapping


def test_f0_stats():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(test_dir, "assets", "txt")
    wav_path = os.path.join(test_dir, "assets", "wav")

    ids, _ = read_txt_and_mapping(txt_path, wav_path, ignore_missing=False)
    assert len(ids) == 10
    evaluator = F0StatsEvaluator(
        ids,
        wav_path,
        ignore_errors=False,
    )
    metrics = evaluator.get_metric()
    assert len(metrics) == 3
    for name, val in metrics:
        assert name.endswith("_std")
        assert val > 0
        print(f"{name}: {val}", flush=True)
