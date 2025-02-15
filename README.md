# speech_gen_eval
Collection of tools to run objective evaluation of speech generation.
Can be used to evaluate various speech generations systems: TTS, Zero-TTS, Zero-VC, Vocoder.

Contains evaluators for:
* Intelligibility with Character Error Rate (CER) via [Whisper Large v3 Turbo](https://huggingface.co/openai/whisper-large-v3-turbo).
* Perceptual quality via [UTMOS](https://huggingface.co/balacoon/utmos) or [UTMOSv2](https://github.com/sarulab-speech/UTMOSv2).
* [Audiobox Aesthetics](https://github.com/facebookresearch/audiobox-aesthetics) 
* Speaker similarity with Speaker Embedding Cosine Similarity (SECS) via [ECAPA](https://huggingface.co/balacoon/ecapa) or [ECAPA v2](https://huggingface.co/Jenthe/ECAPA2) for Zero-TTS and Zero-VC
* Expressiveness via F0, F0 delta, and RMS standard deviation for TTS.
* F0 accuracy (fine and gross errors) and correlation for Vocoders.
* Jitter and Shimmer via [OpenSMILE](https://github.com/audeering/opensmile) for Vocoders.

## Usage

Once you generated synthetic speech for evaluation, run:

```bash
speech-gen-eval --txt <generated text> --generated-audio <dir-with-generated-speech> \
    [--original-audio <dir-with-original-speech>] \
    [--mapping <mapping-between-generated-and-original>] \
    --type <tts|zero-tts|zero-vc|vocoder> \
    --out <output-yaml-file>
```

Where:
* `<generated text>` is a text file of format `id text` for each line.
* `<dir-with-generated-speech>` is a directory with generated speech audio files, named as `<id>.wav`.
* `<dir-with-original-speech>` is a directory with recordings used as references in Zero-TTS or Zero-VC.
* `<mapping-between-generated-and-original>` is a mapping of format `id reference_id` for each line, where `id` is the id of the generated speech and `reference_id` is an id of the audio file from original speech directory used as a reference.
* `<output-yaml-file>` is a yaml file to save the metrics.

Or you can run an underlying python method directly. See `notebooks/xtts.ipynb` for an example how to run an evaluation.

## Installation

Clone the repo, install dependencies, the package and you are good to go.

```bash
# a good idea to create a virtual environment, there are a lot of dependencies
git clone https://github.com/balacoon/speech_gen_eval.git
cd speech_gen_eval
# you need ffmpeg binary in your PATH for audio normalization
apt-get install ffmpeg
pip install -r requirements.txt
pip install .
```

Likely you want to run it on a machine with GPU.
It will work on CPU-only but will be very slow.

## Docker


