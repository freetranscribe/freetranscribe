# freetranscribe

Open tooling from [FreeTranscribe.org](https://freetranscribe.org/). This version has one module, `freetranscribe.parakeet`, which decodes long recordings with NVIDIA Parakeet in [NeMo](https://github.com/NVIDIA-NeMo/Speech): 256-frame local attention, cuts at pauses into pieces of at most 90 seconds, and 5 seconds of overlap between pieces. It works with [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) as released and with [freetranscribe/parakeet-ft-v1](https://huggingface.co/freetranscribe/parakeet-ft-v1), a fine-tune of v3 for long recordings.

The [model card](https://huggingface.co/freetranscribe/parakeet-ft-v1) has the results, how the recipe works, and how I chose and measured it. [`leaderboard/`](leaderboard/README.md) has what you need to reproduce the results.

## Install

With [uv](https://docs.astral.sh/uv/), as a command:

```sh
uv tool install "freetranscribe[nemo]"
```

Or into a project, with `uv add "freetranscribe[nemo]"` or `pip install "freetranscribe[nemo]"`.

It needs Python 3.10 or later. The `nemo` extra pulls in NeMo and PyTorch, which are several gigabytes.

## Usage

```sh
freetranscribe transcribe recording.wav
freetranscribe transcribe recording.wav --model nvidia/parakeet-tdt-0.6b-v3 --json
```

With no options it downloads and loads `freetranscribe/parakeet-ft-v1` on `cuda`. Pass `--model nvidia/parakeet-tdt-0.6b-v3` for v3 as released, or a path to a `.nemo` file, and `--device cpu` if there is no GPU. `--max-seconds`, `--overlap-seconds` and `--batch-size` (default 8) set the recipe. `--json` prints one object per file with the text, each word's start and end time, and the pieces the recording was cut into.

It reads any file soundfile can, such as WAV, FLAC, MP3 and Ogg, averages the channels, and resamples to the 16 kHz the model takes.

From Python:

```python
import soundfile
from freetranscribe import parakeet

asr = parakeet.load_model("freetranscribe/parakeet-ft-v1")
wave, rate = soundfile.read("recording.wav", dtype="float32")
result = parakeet.transcribe(asr, wave, rate)
print(result["text"])
```

At the defaults a batch is eight clips of up to 100 seconds (a 90-second piece plus the overlap on both sides), which took between 11 and 13 GB of GPU memory in my runs. Longer pieces or bigger batches need more.

## Development

```sh
uv run --with ruff ruff check .
uv run python -m unittest discover -s tests
```

The tests cover the cutting, the input handling and the command line, and don't need NeMo or a GPU.

## License and credit

The cutting rule comes from [WhisperX](https://arxiv.org/abs/2303.00747) by Max Bain, Jaesung Huh, Tengda Han and Andrew Zisserman. I wrote the code here from their paper. The models are NVIDIA's Parakeet v3 and a fine-tune of it.

The code is MIT, except `leaderboard/`, which holds a modified copy of the Open ASR Leaderboard's evaluation script under the Apache License 2.0 with its license text beside it. The model weights the tool downloads are not part of this repository: both Parakeet v3 and parakeet-ft-v1 are CC-BY-4.0.
