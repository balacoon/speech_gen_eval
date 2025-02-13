"""
Copyright 2025 Balacoon

Audio directory - utilities for converting audio files
"""

import concurrent.futures
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import resampy
import torch
import torchaudio


def get_audio_path(directory: str, name: str) -> Optional[str]:
    """
    Get the path to an audio file in the given directory
    Args:
        directory (str): The directory to search for the audio file
        name (str): The name of the audio file (without the extension)
    Returns:
        Optional[str]: The path to the audio file if found, otherwise None
    """
    extensions = [".wav", ".mp3", ".flac", ".ogg"]
    file_path = None
    for ext in extensions:
        candidate = os.path.join(directory, name + ext)
        if os.path.exists(candidate):
            file_path = candidate
            break
    return file_path


def _read_audio(directory: str, name: str, sample_rate: int) -> torch.Tensor:
    """
    Read an audio file and return a tensor
    Args:
        directory (str): The directory to search for the audio file
        name (str): The name of the audio file (without the extension)
        sample_rate (int): The sample rate to resample the audio to
        return_torch (bool): Whether to return a torch.Tensor, otherwise return a NumPy array
    Returns:
        torch.Tensor: A tensor containing the audio data
    """
    file_path = get_audio_path(directory, name)
    if file_path is None:
        raise FileNotFoundError(
            f"No supported audio file found for '{name}' in '{directory}'."
        )

    # Get the original sample rate using torchaudio
    info = torchaudio.info(file_path)
    orig_sample_rate = info.sample_rate

    # FFmpeg command to normalize loudness using speechnorm
    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        file_path,
        "-af",
        "speechnorm=e=5:r=0.0003:l=1",  # Apply speech-specific loudness normalization
        "-f",
        "f32le",  # Output raw 32-bit float PCM
        "-ac",
        "1",  # Convert to mono
        "-ar",
        str(orig_sample_rate),  # Keep original sample rate
        "pipe:1",  # Send output to stdout (pipe)
    ]

    # Run FFmpeg and capture output stream
    process = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        bufsize=10**8,
    )
    raw_audio = process.stdout.read()

    # Convert raw PCM data to NumPy array
    waveform = np.frombuffer(raw_audio, dtype=np.float32)

    # Ensure valid audio data
    if waveform.size == 0:
        raise RuntimeError(
            "Failed to decode audio. Check FFmpeg installation and input file."
        )

    # Resample only if necessary
    if orig_sample_rate != sample_rate:
        waveform = resampy.resample(
            waveform, orig_sample_rate, sample_rate, filter="kaiser_best"
        )

    # Convert to PyTorch tensor and return
    return torch.tensor(waveform).unsqueeze(0)  # Add channel dimension


def _convert_audio_file(directory: str, name: str, sample_rate: int, output_dir: str):
    """
    Helper function to read, process, and save a single audio file.
    This function runs in parallel using ProcessPoolExecutor.
    """
    try:
        # Read audio and process it
        audio = _read_audio(directory, name, sample_rate)

        # Define output path
        output_path = Path(output_dir) / f"{name}.wav"

        # Save processed audio
        torchaudio.save(str(output_path), audio, sample_rate, format="wav")

        return name  # Return name of successfully processed file
    except Exception as e:
        print(f"Error processing {name}: {e}")
        return None  # Return None on failure


@contextmanager
def convert_audio_dir(
    directory: str, ids: list[tuple[str, str]], sample_rate: int, njobs: int = 8
):
    """
    Context manager that converts audio files in parallel and stores them in a temporary directory.
    The directory is automatically deleted when the context exits.

    Args:
        directory (str): Input directory containing audio files.
        ids (list[tuple[str, str]]): List of (filename, metadata) tuples.
        sample_rate (int): Target sample rate.
        max_workers (int): Number of parallel workers (default: 4).

    Yields:
        str: Path to the temporary directory containing processed audio files.
    """

    if directory is None:
        yield None
    else:
        # Create a temporary directory
        tmp_dir = tempfile.mkdtemp()
        try:
            # Use ProcessPoolExecutor for parallel processing
            with concurrent.futures.ProcessPoolExecutor(max_workers=njobs) as executor:
                # Submit tasks for parallel execution
                futures = {
                    executor.submit(
                        _convert_audio_file, directory, name, sample_rate, tmp_dir
                    ): name
                    for name, _ in ids
                }

                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    future.result()

            # Yield the temporary directory containing converted files
            yield tmp_dir

        finally:
            # Ensure cleanup when context exits
            shutil.rmtree(tmp_dir)


def get_audio_paths(directory: str, ids: list[tuple[str, str]]) -> list[str]:
    """
    Get the paths to the audio files in the given directory.
    """
    paths = []
    for name, _ in ids:
        path = get_audio_path(directory, name)
        if path is not None:
            paths.append(path)
    return paths


def sort_ids_by_audio_size(
    directory: str, ids: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """
    Sort the ids by the size of the audio files.
    """
    return sorted(
        ids,
        key=lambda x: os.path.getsize(get_audio_path(directory, x[0])),
        reverse=True,
    )
