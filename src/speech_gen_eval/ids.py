"""
Copyright 2025 Balacoon

IDs - utilities for reading and mapping IDs
"""

import logging
import re
from typing import Optional

from speech_gen_eval.audio_dir import get_audio_path


def read_txt_and_mapping(  # noqa: C901
    txt_path: str,
    generated_audio: str,
    mapping_path: Optional[str] = None,
    original_audio: Optional[str] = None,
    ignore_missing: bool = True,
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
    Returns:
        tuple[list[tuple[str, str]], dict[str, str]]: A tuple containing a list of tuples,
        where each tuple contains a name and an utterance, and a dictionary,
        where each key is a name and each value is a reference name
    """
    txt = []
    with open(txt_path, "r") as fp:
        for line in fp:
            name, utterance = re.split(r"\s+", line.strip(), maxsplit=1)
            if get_audio_path(generated_audio, name) is None:
                msg = f"{name} is missing from {generated_audio}, skipping"
                if ignore_missing:
                    logging.warning(msg)
                    continue
                else:
                    raise RuntimeError(msg)
            txt.append((name, utterance))

    if mapping_path is None:
        # no mapping file, if original audio provided, filter ids by original audio too
        if original_audio is not None:
            filt_txt = []
            for name, utterance in txt:
                if get_audio_path(original_audio, name) is None:
                    msg = f"{name} is missing from {original_audio}, skipping"
                    if ignore_missing:
                        logging.warning(msg)
                        continue
                    else:
                        raise RuntimeError(msg)
                filt_txt.append((name, utterance))
            return filt_txt, None
        else:
            return txt, None

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
        if get_audio_path(original_audio, ref_name) is None:
            msg = f"reference {ref_name} is missing from {original_audio}, skipping"
            if ignore_missing:
                logging.warning(msg)
                continue
            else:
                raise RuntimeError(msg)
        filt_txt.append((name, utterance))
        filt_mapping[name] = ref_name
    return filt_txt, filt_mapping
