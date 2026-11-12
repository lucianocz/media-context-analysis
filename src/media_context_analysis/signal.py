"""Frame and audio measurements used by the frozen detector.

Adapted from the narrative-context-segmentation extractor at commit
68cc838a008ad279752bcf598e2039fcabac8be0. Raw frames and PCM are not retained.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray


def hsv_histogram(
    rgb: NDArray[np.uint8], hue_bins: int, sat_bins: int, val_bins: int
) -> NDArray[np.float32]:
    """Normalized HSV histogram. Hue is divided into equal bins over one turn."""
    pixels = rgb.astype(np.float32) * np.float32(1.0 / 255.0)
    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]
    maxc = np.maximum(np.maximum(red, green), blue)
    minc = np.minimum(np.minimum(red, green), blue)
    delta = maxc - minc
    saturation = np.divide(delta, maxc, out=np.zeros_like(maxc), where=maxc > 1e-8)
    hue = np.zeros_like(maxc)
    colored = delta > 1e-8
    red_max = colored & (maxc == red)
    green_max = colored & (maxc == green) & ~red_max
    blue_max = colored & ~red_max & ~green_max
    hue[red_max] = np.mod((green[red_max] - blue[red_max]) / delta[red_max], 6.0)
    hue[green_max] = ((blue[green_max] - red[green_max]) / delta[green_max]) + 2.0
    hue[blue_max] = ((red[blue_max] - green[blue_max]) / delta[blue_max]) + 4.0
    hue *= np.float32(1.0 / 6.0)
    hue_index = np.clip((hue * hue_bins).astype(np.int32), 0, hue_bins - 1)
    sat_index = np.clip((saturation * sat_bins).astype(np.int32), 0, sat_bins - 1)
    val_index = np.clip((maxc * val_bins).astype(np.int32), 0, val_bins - 1)
    flat = (hue_index * (sat_bins * val_bins) + sat_index * val_bins + val_index).ravel()
    histogram = np.bincount(flat, minlength=hue_bins * sat_bins * val_bins).astype(np.float32)
    total = float(histogram.sum())
    if total > 0:
        histogram /= np.float32(total)
    return histogram


def grayscale_thumbnail(rgb: NDArray[np.uint8], thumb_x: int, thumb_y: int) -> NDArray[np.float32]:
    """Unit-norm grayscale thumbnail. Block averages avoid storing the frame."""
    height, width, _channels = rgb.shape
    block_y = height // thumb_y
    block_x = width // thumb_x
    cropped = rgb[: thumb_y * block_y, : thumb_x * block_x].astype(np.float32)
    blocks = cropped.reshape(thumb_y, block_y, thumb_x, block_x, 3).mean(axis=(1, 3))
    gray = 0.2126 * blocks[:, :, 0] + 0.7152 * blocks[:, :, 1] + 0.0722 * blocks[:, :, 2]
    vector = gray.reshape(-1).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-8:
        vector /= np.float32(norm)
    return cast(NDArray[np.float32], vector)


def intersection(left: NDArray[np.floating], right: NDArray[np.floating]) -> float:
    return float(np.minimum(left, right).sum())


def cosine(left: NDArray[np.floating], right: NDArray[np.floating]) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(left, right) / denom)


def frame_change_scores(
    hists: NDArray[np.floating], thumbs: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Blend of histogram distance and thumbnail distance. Index 0 is always 0."""
    scores = np.zeros(len(hists), dtype=np.float64)
    for index in range(1, len(hists)):
        place_distance = 1.0 - intersection(hists[index - 1], hists[index])
        framing_distance = 1.0 - cosine(thumbs[index - 1], thumbs[index])
        scores[index] = 0.65 * place_distance + 0.35 * framing_distance
    return scores


def _band_ratio(
    power: NDArray[np.floating], freqs: NDArray[np.floating], low: float, high: float, total: float
) -> float:
    mask = (freqs >= low) & (freqs < high)
    if total <= 1e-12 or not np.any(mask):
        return 0.0
    return float(power[mask].sum() / total)


def spectral_features(
    samples: NDArray[np.float32],
    sample_rate: int,
    window_s: float,
    hop_s: float,
) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
    """8-D spectral rows and the time at the center of each window.

    The window loop matches the frozen extractor, including the even-length
    Hann window. Callers may stream PCM into this only when the concatenation
    equals the full buffer; ``acoustic_features_from_pcm`` is that path.
    """
    window = int(window_s * sample_rate)
    hop = int(hop_s * sample_rate)
    if window % 2:
        window += 1
    if window <= 0 or hop <= 0 or samples.size < window:
        return np.zeros(0, dtype=np.float64), np.zeros((0, 8), dtype=np.float32)
    hann = np.hanning(window).astype(np.float32)
    freqs = np.fft.rfftfreq(window, d=1.0 / sample_rate)
    rows: list[list[float]] = []
    times: list[float] = []
    nyquist = sample_rate / 2.0
    for start in range(0, samples.size - window + 1, hop):
        chunk = samples[start : start + window]
        weighted = chunk * hann
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        signs = np.sign(chunk)
        signs[signs == 0] = 1
        zero_cross = float(np.mean(np.abs(np.diff(signs))) * 0.5)
        spectrum = np.fft.rfft(weighted)
        power = (spectrum.real**2 + spectrum.imag**2).astype(np.float64)
        total = float(power.sum()) + 1e-12
        centroid = float(np.sum(freqs * power) / total)
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total))
        cumulative = np.cumsum(power) / total
        rolloff_index = int(np.searchsorted(cumulative, 0.85))
        rolloff_index = min(rolloff_index, len(freqs) - 1)
        rolloff = float(freqs[rolloff_index])
        flatness = float(np.exp(np.mean(np.log(power + 1e-12))) / (np.mean(power) + 1e-12))
        rows.append(
            [
                float(np.log(rms + 1e-8)),
                zero_cross,
                centroid / nyquist,
                bandwidth / nyquist,
                rolloff / nyquist,
                flatness,
                _band_ratio(power, freqs, 0.0, 500.0, total),
                _band_ratio(power, freqs, 4000.0, nyquist, total),
            ]
        )
        times.append((start + window / 2.0) / sample_rate)
    return np.asarray(times, dtype=np.float64), np.asarray(rows, dtype=np.float32)


class PcmFeatureReducer:
    """Streaming equivalent of :func:`spectral_features` on one contiguous PCM buffer.

    Overlapping windows are formed by dropping ``hop`` samples after each row.
    The numeric result matches calling :func:`spectral_features` on the joined samples.
    """

    def __init__(self, sample_rate: int, window_s: float, hop_s: float) -> None:
        self.sample_rate = sample_rate
        self.window = int(window_s * sample_rate)
        self.hop = int(hop_s * sample_rate)
        if self.window % 2:
            self.window += 1
        self._pending = np.empty(0, dtype=np.float32)
        self._origin = 0
        self._rows: list[list[float]] = []
        self._times: list[float] = []
        self._closed = False
        if self.window > 0 and self.hop > 0:
            self._hann = np.hanning(self.window).astype(np.float32)
            self._freqs = np.fft.rfftfreq(self.window, d=1.0 / sample_rate)
        else:
            self._hann = np.empty(0, dtype=np.float32)
            self._freqs = np.empty(0, dtype=np.float64)

    def extend(self, samples: NDArray[np.float32]) -> None:
        if self._closed or samples.size == 0 or self.window <= 0 or self.hop <= 0:
            return
        if self._pending.size == 0:
            self._pending = samples.astype(np.float32, copy=True)
        else:
            self._pending = np.concatenate([self._pending, samples.astype(np.float32, copy=False)])
        self._drain()

    def finish(self) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
        self._closed = True
        if not self._times:
            return np.zeros(0, dtype=np.float64), np.zeros((0, 8), dtype=np.float32)
        return np.asarray(self._times, dtype=np.float64), np.asarray(self._rows, dtype=np.float32)

    def _drain(self) -> None:
        nyquist = self.sample_rate / 2.0
        while self._pending.size >= self.window:
            chunk = self._pending[: self.window]
            weighted = chunk * self._hann
            rms = float(np.sqrt(np.mean(np.square(chunk))))
            signs = np.sign(chunk)
            signs[signs == 0] = 1
            zero_cross = float(np.mean(np.abs(np.diff(signs))) * 0.5)
            spectrum = np.fft.rfft(weighted)
            power = (spectrum.real**2 + spectrum.imag**2).astype(np.float64)
            total = float(power.sum()) + 1e-12
            centroid = float(np.sum(self._freqs * power) / total)
            bandwidth = float(np.sqrt(np.sum(((self._freqs - centroid) ** 2) * power) / total))
            cumulative = np.cumsum(power) / total
            rolloff_index = min(int(np.searchsorted(cumulative, 0.85)), len(self._freqs) - 1)
            rolloff = float(self._freqs[rolloff_index])
            flatness = float(np.exp(np.mean(np.log(power + 1e-12))) / (np.mean(power) + 1e-12))
            self._rows.append(
                [
                    float(np.log(rms + 1e-8)),
                    zero_cross,
                    centroid / nyquist,
                    bandwidth / nyquist,
                    rolloff / nyquist,
                    flatness,
                    _band_ratio(power, self._freqs, 0.0, 500.0, total),
                    _band_ratio(power, self._freqs, 4000.0, nyquist, total),
                ]
            )
            self._times.append((self._origin + self.window / 2.0) / self.sample_rate)
            self._pending = self._pending[self.hop :]
            self._origin += self.hop


def acoustic_change_scores(
    features: NDArray[np.floating], hop_s: float, window_s: float
) -> NDArray[np.float64]:
    """Cosine distance between mean descriptors of the previous and next window.

    A full covariance BIC is unstable at this feature size and window length.
    The cosine distance is the inexpensive stand-in for that change statistic.
    """
    count = int(features.shape[0]) if features.ndim == 2 else 0
    scores = np.zeros(count, dtype=np.float64)
    if count == 0:
        return scores
    half = max(1, round(window_s / hop_s))
    median = np.median(features, axis=0)
    quartile_low = np.percentile(features, 25, axis=0, method="linear")
    quartile_high = np.percentile(features, 75, axis=0, method="linear")
    scale = np.maximum(quartile_high - quartile_low, 1e-3)
    scaled = (features - median) / scale
    for index in range(half, count - half):
        left = scaled[index - half : index].mean(axis=0)
        right = scaled[index : index + half].mean(axis=0)
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm < 1e-8 or right_norm < 1e-8:
            cosine_value = 1.0
        else:
            cosine_value = float(np.dot(left, right) / (left_norm * right_norm))
        scores[index] = max(0.0, 1.0 - cosine_value)
    return scores
