"""Project token embeddings to 2D so the Fourier ring is visible, and align
frames so the ring glides rather than flips between adjacent checkpoints.

Nanda et al. 2023 shows the grokked model encodes tokens as points on a
circle at angle 2πkt/p for some dominant frequency k. Top-2 PCA mixes the
4–5 significant frequencies and washes the ring out — we instead project
onto the cos/sin basis for the single dominant frequency of the final
embedding. That same basis is applied to every earlier checkpoint so the
transition from scatter → ring is visible."""

import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes


def dominant_frequency(embeds_final: np.ndarray) -> int:
    """Return the k in [1, p/2] with the most power across embedding dims."""
    p = embeds_final.shape[0]
    ft = np.fft.fft(embeds_final, axis=0)
    power = (np.abs(ft) ** 2).sum(axis=1)
    half = p // 2 + 1
    return int(np.argmax(power[1:half]) + 1)


def fourier_basis(d: int, p: int, k: int, embeds_final: np.ndarray) -> np.ndarray:
    """Return the 2 d-dim directions (cos_dir, sin_dir) that isolate frequency
    k in the final embedding. Built from the FFT of the final embedding so
    they reflect where the ring ends up."""
    t = np.arange(p)
    cos_t = np.cos(2 * np.pi * k * t / p)
    sin_t = np.sin(2 * np.pi * k * t / p)
    cos_dir = cos_t @ embeds_final  # (d,)
    sin_dir = sin_t @ embeds_final  # (d,)

    basis = np.stack([cos_dir, sin_dir], axis=0)
    basis = basis / np.linalg.norm(basis, axis=1, keepdims=True)
    return basis.astype(np.float32)  # (2, d)


def project(embeds: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project every frame's embeddings onto the shared 2D basis."""
    return embeds @ basis.T  # (n, p, 2)


def canonicalize_orientation(points: np.ndarray) -> np.ndarray:
    """Put token 0 on the +x axis so the ring starts from a stable angle."""
    angle = np.arctan2(points[0, 1], points[0, 0])
    c, s = np.cos(-angle), np.sin(-angle)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return points @ rot.T


def align_sequence(projs: np.ndarray) -> np.ndarray:
    """Align each frame to the previous via Procrustes. First frame is
    canonicalized so the sequence starts from a stable orientation."""
    n, p, _ = projs.shape
    aligned = np.empty_like(projs)
    aligned[0] = canonicalize_orientation(projs[0])
    for i in range(1, n):
        r, _ = orthogonal_procrustes(projs[i], aligned[i - 1])
        aligned[i] = projs[i] @ r
    return aligned


def align_arrays(arrays_path: Path, out_path: Path) -> None:
    data = np.load(arrays_path)
    embeds = data["embeds"]  # (n, p, d)
    n, p, d = embeds.shape

    final = embeds[-1]
    k = dominant_frequency(final)
    basis = fourier_basis(d, p, k, final)
    print(f"dominant frequency: k={k} ({k}/{p} = {k/p:.3f})")

    projs = project(embeds.astype(np.float32), basis)
    aligned = align_sequence(projs)

    out = {k_: data[k_] for k_ in data.files}
    out["embeds_2d"] = aligned.astype(np.float32)
    out["fourier_k"] = np.array(k, dtype=np.int32)
    np.savez_compressed(out_path, **out)
    print(f"wrote aligned arrays to {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", default="checkpoints/raw/arrays.npz", type=Path)
    ap.add_argument("--out", default="checkpoints/raw/arrays_aligned.npz", type=Path)
    args = ap.parse_args()
    align_arrays(args.arrays, args.out)


if __name__ == "__main__":
    main()
