"""
Copyright 2025 Balacoon

IDs - utilities for reading and mapping IDs
"""

import logging
import re
from typing import Optional

import soundfile as sf

from speech_gen_eval.audio_dir import get_audio_path


def _is_audio_good(
    directory: str, name: str, ignore_missing: bool, min_dur: float, max_dur: float
) -> bool:
    """
    Check if an audio file is good
    """
    path = get_audio_path(directory, name)
    if path is None:
        msg = f"Skipping {name} because it is missing from {directory}"
        if not ignore_missing:
            raise ValueError(msg)
        logging.warning(msg)
        return False
    info = sf.info(path)
    duration = info.frames / info.samplerate
    if duration < min_dur or duration > max_dur:
        msg = f"Skipping {name} because of duration {duration} (min: {min_dur}, max: {max_dur})"
        if not ignore_missing:
            raise ValueError(msg)
        logging.warning(msg)
        return False
    return True


def read_txt_and_mapping(  # noqa: C901
    txt_path: str,
    generated_audio: str,
    mapping_path: Optional[str] = None,
    original_audio: Optional[str] = None,
    ignore_missing: bool = True,
    min_dur: float = 0.3,
    max_dur: float = 40.0,
) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """
    Read a text file and a mapping file, and return a list of tuples,
    where each tuple contains a name and an utterance, and a dictionary,
    where each key is a name and each value is a reference name
    Args:
        txt_path (str): The path to the text file
        generated_audio (str): The directory to search for the audio files
        mapping_path (Optional[str]): The path to the mapping file
        original_audio (Optional[str]): The directory to search for the original audio files
        ignore_missing (bool): Whether to ignore missing files
        min_dur (float): The minimum duration of the audio files to consider (default: 0.3s).
        max_dur (float): The maximum duration of the audio files to consider (default: 40.0s).
    Returns:
        tuple[list[tuple[str, str]], dict[str, str]]: A tuple containing a list of tuples,
        where each tuple contains a name and an utterance, and a dictionary,
        where each key is a name and each value is a reference name
    """
    txt = []
    with open(txt_path, "r", encoding="utf-8") as fp:
        for line in fp:
            name, utterance = re.split(r"\s+", line.strip(), maxsplit=1)
            if _is_audio_good(generated_audio, name, ignore_missing, min_dur, max_dur):
                txt.append((name, utterance))

    if mapping_path is None:
        # no mapping file, if original audio provided, filter ids by original audio too
        if original_audio is not None:
            filt_txt = []
            for name, utterance in txt:
                if _is_audio_good(
                    original_audio, name, ignore_missing, min_dur, max_dur
                ):
                    filt_txt.append((name, utterance))
            return filt_txt, None
        else:
            return txt, None
    else:
        if original_audio is None:
            raise ValueError("original_audio is required when mapping_path is provided")
    # read mapping file
    mapping = {}
    with open(mapping_path, "r") as fp:
        for line in fp:
            name, ref = line.strip().split()
            mapping[name] = ref

    filt_txt = []
    filt_mapping = {}
    for name, utterance in txt:
        if name not in mapping:
            msg = f"mapping for {name} is missing from {mapping_path}"
            if ignore_missing:
                logging.warning(msg)
                continue
            else:
                raise RuntimeError(msg)
        ref_name = mapping[name]
        if _is_audio_good(original_audio, ref_name, ignore_missing, min_dur, max_dur):
            filt_txt.append((name, utterance))
            filt_mapping[name] = ref_name
    return filt_txt, filt_mapping
