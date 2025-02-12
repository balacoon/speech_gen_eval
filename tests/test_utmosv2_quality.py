"""
Copyright 2025 Balacoon

Test - test the UTMOSv2quality evaluator
"""

import os

from speech_gen_eval.ids import read_txt_and_mapping
from speech_gen_eval.utmosv2_quality import UTMOSv2QualityEvaluator


def test_quality():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(test_dir, "assets", "txt")
    wav_path = os.path.join(test_dir, "assets", "wav")

    ids, _ = read_txt_and_mapping(txt_path, wav_path, ignore_missing=False)
    assert len(ids) == 10
    evaluator = UTMOSv2QualityEvaluator(
        ids,
        wav_path,
        ignore_errors=False,
    )
    metrics = evaluator.get_metric()
    assert len(metrics) == 1
    name, val = metrics[0]
    assert name == "utmosv2_mos"
    assert val > 2.5
