"""Choose where to cut a long recording so that each piece ends in a pause.

The pause detector is short-time log energy, scaled between the recording's own noise floor and
speech level. The cutting rule is the min-cut of WhisperX (Bain et al., 2023,
https://arxiv.org/abs/2303.00747) applied to the whole recording: each cut falls on the lowest
score in the second half of the allowed piece length, and no audio is dropped.
"""

import numpy as np

HOP_SECONDS = 0.02


def smooth(values: np.ndarray, seconds: float) -> np.ndarray:
    """Moving average over `seconds`, same length as the input."""
    width = max(1, round(seconds / HOP_SECONDS))
    return np.convolve(values, np.ones(width) / width, mode="same")


def energy_scores(wave: np.ndarray, rate: int) -> np.ndarray:
    """One speech-activity score in [0, 1] per 20 ms frame.

    Frame power in decibels is scaled so the recording's 5th percentile maps to 0 and its 95th to
    1, then smoothed over 200 ms so a cut lands inside a pause instead of on a brief dip between
    two words.
    """
    hop = round(HOP_SECONDS * rate)
    frames = len(wave) // hop
    power = (wave[: frames * hop].astype(np.float64) ** 2).reshape(frames, hop).mean(axis=1)
    decibels = 10 * np.log10(power + 1e-10)
    floor, level = np.percentile(decibels, [5, 95])
    return smooth(np.clip((decibels - floor) / max(level - floor, 1e-6), 0, 1), 0.2)


def mincut(scores: np.ndarray, max_seconds: float) -> list[tuple[float, float]]:
    """Cut a recording into consecutive pieces of at most `max_seconds`.

    Returns (start, end) times in seconds that cover the whole recording. Each cut is placed at
    the lowest score between half and all of the allowed length after the previous cut.
    """
    size = round(max_seconds / HOP_SECONDS)
    start, end, pieces = 0, len(scores), []
    while end - start > size:
        low, high = start + size // 2, start + size
        cut = low + int(np.argmin(scores[low:high]))
        pieces.append((start, cut))
        start = cut
    if end > start:
        pieces.append((start, end))
    return [(begin * HOP_SECONDS, finish * HOP_SECONDS) for begin, finish in pieces]
