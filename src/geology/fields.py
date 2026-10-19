"""Spatially correlated Gaussian random fields (spectral filtering)."""
from __future__ import annotations

import numpy as np


def gaussian_random_field(rng: np.random.Generator, shape: tuple[int, ...], spacing: tuple[float, ...],
                          corr_lengths: tuple[float, ...]) -> np.ndarray:
    """Zero-mean, unit-variance anisotropic Gaussian-covariance random field.

    The domain is padded by ~3 correlation lengths before FFT filtering so the
    periodic wrap-around of the FFT does not correlate opposite boundaries.
    """
    pad = [int(np.ceil(3 * L / d)) for L, d in zip(corr_lengths, spacing)]
    full = [n + p for n, p in zip(shape, pad)]
    noise = rng.standard_normal(full)
    filt = np.ones(full)
    for ax, (n, d, L) in enumerate(zip(full, spacing, corr_lengths)):
        k = np.fft.fftfreq(n, d=d)
        f1 = np.exp(-0.5 * (2 * np.pi * k * L / 2.0) ** 2)
        sh = [1] * len(full)
        sh[ax] = n
        filt = filt * f1.reshape(sh)
    field = np.real(np.fft.ifftn(np.fft.fftn(noise) * filt))
    field = field[tuple(slice(0, n) for n in shape)]
    field -= field.mean()
    std = field.std()
    return field / (std if std > 0 else 1.0)


def gaussian_blur_2d(img: np.ndarray, sigma_cells: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(img, sigma=sigma_cells, mode="nearest")
