"""Decode a long recording in pause-cut pieces with overlap.

Each piece is decoded with extra audio on both sides. Only the words whose midpoints fall inside
the piece are kept, so every word comes from a decode that saw context on both sides of it.
"""

from typing import Any

import numpy as np

from .segment import energy_scores, mincut

MAX_SECONDS = 90.0
OVERLAP_SECONDS = 5.0
ATTENTION_CONTEXT = (256, 256)


def load_model(model: str, device: str = "cuda") -> Any:
    """Load a NeMo Parakeet model from a `.nemo` file or a Hugging Face model id.

    The model is switched to local attention with 256 frames of context on each side and, on a
    GPU, cast to BF16 before it is moved there.
    """
    try:
        import torch
        from nemo.collections.asr.models import ASRModel
        from omegaconf import open_dict
    except ImportError as error:
        raise ImportError(
            'Parakeet decoding needs NeMo and PyTorch. Install them with the nemo extra, '
            'for example: uv tool install "freetranscribe[nemo]"'
        ) from error

    loader = ASRModel.restore_from if model.endswith(".nemo") else ASRModel.from_pretrained
    asr = loader(model, map_location="cpu")
    asr.change_attention_model("rel_pos_local_attn", list(ATTENTION_CONTEXT))
    if device != "cpu":
        asr.to(torch.bfloat16)
    asr.to(device)
    asr.eval()
    if asr.cfg.decoding.strategy != "beam":
        with open_dict(asr.cfg.decoding):
            asr.cfg.decoding.strategy = "greedy_batch"
            # NeMo's CUDA-graph greedy decoder stops with an illegal memory access on some
            # recordings, so decode with the plain one.
            asr.cfg.decoding.greedy.use_cuda_graph_decoder = False
        asr.change_decoding_strategy(asr.cfg.decoding, verbose=False)
    return asr


def mono_16k(wave: np.ndarray, rate: int) -> np.ndarray:
    """Average the channels of `wave` and resample it to 16 kHz."""
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if rate != 16000:
        import soxr

        wave = soxr.resample(wave, rate, 16000)
    return wave.astype(np.float32, copy=False)


def transcribe(asr: Any, wave: np.ndarray, rate: int = 16000, *, max_seconds: float = MAX_SECONDS,
               overlap_seconds: float = OVERLAP_SECONDS, batch_size: int = 8) -> dict[str, Any]:
    """Transcribe one recording.

    `wave` is float samples, mono or with one column per channel; channels are averaged. The
    model takes 16 kHz audio, so any other sample rate is resampled to it first.
    Returns the joined text, the words with start and end times in seconds from the start of the
    recording, and the pieces the recording was cut into.
    """
    wave, rate = mono_16k(wave, rate), 16000

    import torch

    total = len(wave) / rate
    pieces = mincut(energy_scores(wave, rate), max_seconds)
    padded = [(max(0.0, begin - overlap_seconds), min(total, finish + overlap_seconds))
              for begin, finish in pieces]
    clips = [wave[round(begin * rate):round(finish * rate)] for begin, finish in padded]
    order = sorted(range(len(clips)), key=lambda index: -len(clips[index]))
    kept: list[list[dict[str, Any]]] = [[] for _ in clips]
    with torch.inference_mode():
        for offset in range(0, len(order), batch_size):
            batch = order[offset:offset + batch_size]
            outputs = asr.transcribe([clips[index] for index in batch], batch_size=len(batch),
                                     verbose=False, timestamps=True)
            for index, output in zip(batch, outputs, strict=True):
                shift = padded[index][0]
                begin, finish = pieces[index]
                last = index == len(pieces) - 1
                for word in output.timestamp["word"]:
                    middle = shift + (word["start"] + word["end"]) / 2
                    if begin <= middle and (last or middle < finish):
                        kept[index].append({"word": word["word"], "start": shift + word["start"],
                                            "end": shift + word["end"]})
    words = [word for piece in kept for word in piece]
    return {"text": " ".join(word["word"] for word in words), "words": words, "pieces": pieces}
