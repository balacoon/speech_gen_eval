"""
Copyright 2025 Balacoon

Test - test the UTMOSv2quality evaluator
"""

import os

import torch

from speech_gen_eval.ids import read_txt_and_mapping
from speech_gen_eval.utmos_quality import UTMOSQualityEvaluator


def test_quality():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(test_dir, "assets", "txt")
    wav_path = os.path.join(test_dir, "assets", "wav")

    ids, _ = read_txt_and_mapping(txt_path, wav_path, ignore_missing=False)
    assert len(ids) == 10
    evaluator = UTMOSQualityEvaluator(
        ids,
        wav_path,
        ignore_errors=False,
    )
    metrics = evaluator.get_metric()
    if torch.cuda.is_available():
        assert len(metrics) == 1
        name, val = metrics[0]
        assert name == "utmos_mos"
        assert val > 2.5
    else:
        assert len(metrics) == 0
