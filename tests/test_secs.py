"""
Copyright 2025 Balacoon

Test - test the ECAPA2-secs evaluator
"""

import os

import numpy as np
import pytest

from speech_gen_eval.ids import read_txt_and_mapping
from speech_gen_eval.secs import (
    ECAPA2SECSEvaluator,
    ECAPASECSEvaluator,
    ReDimNetSECSEvaluator,
)


@pytest.mark.parametrize(
    "evaluator_cls", [ECAPASECSEvaluator, ECAPA2SECSEvaluator, ReDimNetSECSEvaluator]
)
def test_quality(evaluator_cls):
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

    # for mapping - speakers are different, secs < 0.5
    def real_mapping_check(val_):
        assert val_ < 0.5

    # for fake_mapping - speakers are the same, secs == 1.0
    def fake_mapping_check(val_):
        assert np.abs(val_ - 1.0) < 0.01

    for this_mapping, check_secs in zip(
        [mapping, fake_mapping], [real_mapping_check, fake_mapping_check]
    ):
        assert len(ids) == 10
        assert len(this_mapping) == 10
        evaluator = evaluator_cls(
            ids,
            generated_audio=wav_path,
            mapping=this_mapping,
            original_audio=wav_path,
            ignore_errors=False,
        )
        metrics = evaluator.get_metric()
        assert len(metrics) == 1
        name, val = metrics[0]
        assert name == "ecapa_secs" or name == "ecapa2_secs" or name == "redimnet_secs"
        check_secs(val)
