"""
Copyright 2025 Balacoon

Test - test the ECAPA2-secs evaluator
"""

import os

import numpy as np

from speech_gen_eval.ecapa2_secs import ECAPA2SECSEvaluator
from speech_gen_eval.ids import read_txt_and_mapping


def test_quality():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(test_dir, "assets", "txt")
    mapping_path = os.path.join(test_dir, "assets", "mapping")
    wav_path = os.path.join(test_dir, "assets", "wav")

    ids, mapping = read_txt_and_mapping(
        txt_path,
        wav_path,
        mapping_path=mapping_path,
        original_audio=wav_path,
        ignore_missing=False,
    )
    fake_mapping = {name: name for name, _ in ids}

    # secs == 1.0 - those are the same speakers
    # secs < 0.5 - those are completely different speakers
    for this_mapping, expected_secs in zip([mapping, fake_mapping], [0.4, 1.0]):
        assert len(ids) == 10
        assert len(this_mapping) == 10
        evaluator = ECAPA2SECSEvaluator(
            ids,
            generated_audio=wav_path,
            mapping=this_mapping,
            original_audio=wav_path,
            ignore_errors=False,
        )
        metrics = evaluator.get_metric()
        assert len(metrics) == 1
        name, val = metrics[0]
        assert name == "ecapa2_secs"
        assert np.abs(val - expected_secs) < 0.01
