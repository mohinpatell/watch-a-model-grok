"""Mechanistic progress measures computed from the captured embeddings.

The grokking story is that a Fourier circuit forms during the apparent
plateau, long before test loss falls. We verify this by tracking how much
of the token-embedding energy sits on each Fourier frequency over training.

For each checkpoint, we compute

    power[k] = sum_d |fft(E_d)|_k^2       (d = embedding dim, k = freq)

and normalize by the total (k >= 1) power. The dominant frequency of the
final embedding (k_star = 46 for the seed=42 run) climbs from the floor
~1/(p/2) during memorization to a large share during grokking. Plotting
power[k_star] vs. step is a "did the circuit form?" curve that anticipates
the test-loss snap by thousands of steps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def fft_power_per_frame(embeds: np.ndarray) -> np.ndarray:
    """Return (n_frames, p // 2) FFT power summed across embedding dims,
    dropping the DC component. embeds shape: (n_frames, p, d_model)."""
    n, p, _ = embeds.shape
    ft = np.fft.fft(embeds, axis=1)
    power = (np.abs(ft) ** 2).sum(axis=-1)  # (n, p)
    half = p // 2 + 1
    return power[:, 1:half].astype(np.float32)  # drop DC


def normalized_share(power: np.ndarray) -> np.ndarray:
    """Normalize each row so frequencies sum to 1 (excluding DC)."""
    totals = power.sum(axis=-1, keepdims=True)
    return (power / np.maximum(totals, 1e-12)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arrays",
        default="checkpoints/raw/arrays_aligned.npz",
        type=Path,
    )
    ap.add_argument(
        "--out",
        default="checkpoints/raw/arrays_aligned.npz",
        type=Path,
    )
    args = ap.parse_args()

    data = np.load(args.arrays)
    embeds = data["embeds"]  # (n, p, d)
    k_star = int(data["fourier_k"])

    power = fft_power_per_frame(embeds)
    share = normalized_share(power)
    share_at_k = share[:, k_star - 1]  # k=1..p/2 stored 0-indexed.
    top5_idx = np.argsort(-share[-1])[:5] + 1
    print(f"k_star = {k_star}, top-5 final freqs: {top5_idx.tolist()}")
    print(
        f"share[k*] at final step = {share_at_k[-1]:.3f} "
        f"(baseline 1/(p/2) = {1/(113//2):.3f})"
    )

    out = {k_: data[k_] for k_ in data.files}
    out["fft_share"] = share          # (n, p/2) row-normalized
    out["share_at_kstar"] = share_at_k.astype(np.float32)
    np.savez_compressed(args.out, **out)
    print(f"wrote fft_share, share_at_kstar to {args.out}")


if __name__ == "__main__":
    main()
