---
license: cc-by-4.0
language:
- en
library_name: nemo
pipeline_tag: automatic-speech-recognition
base_model: nvidia/parakeet-tdt-0.6b-v3
base_model_relation: finetune
tags:
- parakeet
- tdt
- speech-recognition
- long-form
---

# parakeet-ft-v1

A fine-tune of [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) for long recordings. v3 sometimes skips whole passages of a long recording, and this model almost never does: on 16 recordings kept out of training, v3 skipped 87 passages and this model 3. Same architecture, same tokenizer, same 0.6B parameters, loadable with [NeMo](https://github.com/NVIDIA-NeMo/Speech) like the original. It loads with the local attention it was trained with, so long recordings fit in memory without extra settings. [FreeTranscribe.org](https://freetranscribe.org/) uses it for every transcription.

It was trained only on its own output. No person wrote or corrected a transcript, and no other speech recognizer was involved.

## Self-training harness

1. Find a failure and build a harness around the model that repairs it. Here the harness finds passages the model skipped and gets the model to transcribe them.
2. Run the model with its harness over a lot of unlabeled audio. The transcripts are better than the model produces alone, and every word in them still comes from the model.
3. Train the model on those transcripts. It no longer needs the harness for that failure.

## Results

With the recipe, this model ranks third of 13 on the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)'s long-form test with a 9.81% average word error rate (WER), and first among models with open weights. Only ElevenLabs Scribe v2 (8.44%) and Reson8 resonant-1 (9.74%) score lower, as of 2026-10-26. The table gives WER in percent, lower is better. "Leaderboard decoding" is the leaderboard's own NeMo script, which decodes the whole recording in one call. "Recipe" is a decoding recipe for long recordings published at [freetranscribe/freetranscribe](https://github.com/freetranscribe/freetranscribe), which also works on v3 without fine-tuning.

| Dataset | v3, leaderboard decoding | v3, recipe | This model, leaderboard decoding | This model, recipe |
| --- | ---: | ---: | ---: | ---: |
| Earnings-21 | 9.40 | 8.11 | 7.62 | 6.97 |
| Earnings-22 (development set) | 13.75 | 11.14 | 10.82 | 10.13 |
| CORAAL, mean of 8 subsets | 14.48 | 13.08 | 13.21 | 12.33 |
| TED-LIUM | 2.47 | 2.29 | 1.95 | 1.95 |
| Mean of Earnings-21, Earnings-22 and CORAAL, as the leaderboard ranks | 12.54 | 10.78 | 10.55 | 9.81 |
| Mean of the three held-out sets: Earnings-21, CORAAL and TED-LIUM | 8.78 | 7.83 | 7.59 | 7.08 |

The recipe and the fine-tuning stack: this model with the recipe is best or tied in every row.

## Usage

With plain NeMo:

```python
import nemo.collections.asr as nemo_asr

asr = nemo_asr.models.ASRModel.from_pretrained("freetranscribe/parakeet-ft-v1")
print(asr.transcribe(["recording.wav"])[0].text)
```

With the recipe, for long recordings:

```sh
uv tool install "freetranscribe[nemo]"
freetranscribe transcribe recording.wav
```

## Where it is weaker

- Short clips are mixed against v3: better on podcasts, talks and noisy speech, worse on meetings and parliamentary speech, and slightly worse on average. If short clips are your main use, try both.
- It was trained on English. Some other languages got worse, so use v3 for those.
- It drops fewer words. It doesn't hear them better, so names and jargon are no more accurate than with v3.

## How I trained and tested it

### Training

The training set was 100 hours of clips from 809 long YouTube recordings, mostly interviews, podcasts and talks in English. Training took about 18 hours on one RTX 3090. Most of the gain arrives early: one hour of data gets about two thirds of the way. None of the training audio was taken from an Open ASR Leaderboard dataset.

### Skipped passages

When v3 skips, the speech is clear. v3 emits nothing for 10, 50 or 250 words and then carries on. I found this on my own jobs: long YouTube interviews and podcasts I submitted to FreeTranscribe.org.

I kept 16 of those recordings (29.6 hours) out of training. The table counts passages of 10 or more consecutive words that are in YouTube's automatic captions and missing from the transcript. Both models were decoded with the recipe.

| Model | Skipped passages | Words skipped | Recordings affected |
| --- | ---: | ---: | ---: |
| Parakeet v3 | 87 | 1,775 | 13 of 16 |
| parakeet-ft-v1 | 3 | 31 | 3 of 16 |

On these recordings the fine-tuned model is what removes the skipped passages. The recipe alone leaves v3 with most of them.

### Decoding recipe

The recipe does three things differently from a plain whole-recording `transcribe` call:

1. It uses local attention with 256 encoder frames of context on each side, about 20 seconds, where the leaderboard's long-audio setting is 128.
2. It cuts the recording at pauses into pieces of at most 90 seconds and decodes the pieces in batches. The pause score is short-time energy, and each cut goes at the quietest point in the second half of the allowed length, the min-cut rule from [WhisperX](https://arxiv.org/abs/2303.00747) (Bain et al., 2023).
3. It decodes each piece with 5 seconds of extra audio on both sides and keeps each word only from the piece its midpoint falls inside.

The next table shows what each step adds on Earnings-22 (125 earnings calls, about 985,000 reference words). Each row adds one step to the row above.

| Decoding | v3 | parakeet-ft-v1 |
| --- | ---: | ---: |
| Whole recording, 128-frame attention | 13.75 | 10.82 |
| Plus 256-frame attention | 12.76 | 10.33 |
| Plus cuts at pauses, 90 seconds at most | 11.12 | 10.17 |
| Plus 5 seconds of overlap | 11.14 | 10.13 |

Wider attention and cutting help both models, and cutting helps v3 far more. The overlap adds little: under 0.1 points on the fine-tuned model and nothing on v3. This model loads with 256-frame attention already. On v3 it is one line in plain NeMo: `asr.change_attention_model("rel_pos_local_attn", [256, 256])`.

I chose the settings on Earnings-22 with the fine-tuned model and 256-frame attention, without overlap unless stated:

- Piece length: 20 seconds scored 10.76, 30 seconds 10.49, 45 seconds 10.32 and 60 seconds 10.22. From 90 to 600 seconds the scores tie between 10.10 and 10.19, so I picked 90, which needs the least memory. The 30-second habit comes from Whisper, which can't take more.
- Pause detector: energy, WebRTC's recurrent-network detector and its older Gaussian-mixture detector scored 10.49, 10.45 and 10.55 at 30 seconds. I kept energy, which needs no extra dependencies.
- Cutting blind: fixed 30-second windows scored 10.69, worse than cutting at pauses on 113 of 125 calls.
- Dropping silence: the full "cut and merge" from WhisperX, which discards audio between speech regions, scored 10.63 at 30 seconds. I tried one threshold setting.
- Overlap: 2 seconds scored 10.15, 5 seconds 10.08 to 10.13 over four runs, and 10 seconds 10.05 in one run. The differences are about the size of the run-to-run spread.

Recordings of 90 seconds or less are not cut, so only the wider attention applies to them. The recipe runs at about half the speed of decoding a recording whole.

### Comparing with other numbers

- Nothing was trained on any of these sets. I used Earnings-22 only to choose between checkpoints and recipe settings. Earnings-21, CORAAL and TED-LIUM played no part in any choice.
- The leaderboard's own run of v3 is within a third of a point of mine on each set, 12.60 against 12.54 on average. Compare the table's columns with each other rather than with other people's runs.
- Scores with the recipe repeat to within a few hundredths of a point from run to run: four runs of this model on Earnings-22 gave 10.08, 10.10, 10.11 and 10.13. v3 decoded whole varies by a few tenths.
- I measured everything with bfloat16 weights and greedy decoding, and scored with the leaderboard's current scorer.
- The repository's [`leaderboard/`](https://github.com/freetranscribe/freetranscribe/tree/main/leaderboard) directory has the evaluation script and the dataset revisions.

## License and credit

CC-BY-4.0, the same as the base model. This is a derivative of [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) by NVIDIA, which did almost all of the work.

This model learned one thing from the harness behind [FreeTranscribe.org](https://freetranscribe.org/). With this model the harness added almost nothing, so the service now runs the model on its own.
