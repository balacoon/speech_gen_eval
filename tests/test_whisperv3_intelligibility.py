"""
Copyright 2025 Balacoon

Test - test the WhisperV3intelligibility evaluator
"""

import os
from speech_gen_eval.whisperv3_intelligibility import WhisperV3IntelligibilityEvaluator
from speech_gen_eval.ids import read_txt_and_mapping


def test_intelligibility():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(test_dir, "assets", "txt")
    wav_path = os.path.join(test_dir, "assets", "wav")

    ids, _ = read_txt_and_mapping(txt_path, wav_path, ignore_missing=False)
    assert len(ids) == 10
    evaluator = WhisperV3IntelligibilityEvaluator(
        ids,
        wav_path,
        ignore_errors=False,
    )
    metrics = evaluator.get_metric()
    assert len(metrics) == 1
    name, val = metrics[0]
    assert name == "whisperv3_cer"
    assert val < 0.02