"""Convert the aligned arrays.npz + summary.json into web-friendly assets:

- meta.json      — per-step scalars (steps, losses, accs) + config + shapes
- embeds.bin     — float32 (n_frames, p, 2) embedding positions
- attn.bin       — float32 (n_frames, probe_batch, n_heads, seq, seq) attention
- fft_share.bin  — float32 (n_frames, p//2) per-frequency embedding power share

Binaries are raw float32 little-endian, loaded in the browser with
`new Float32Array(await res.arrayBuffer())` and reshaped in JS. JSON holds
shapes so the frontend stays decoupled from array layout."""

import argparse
import json
from pathlib import Path

import numpy as np


def export_web(arrays_path: Path, summary_path: Path, out_dir: Path) -> None:
    data = np.load(arrays_path)
    with open(summary_path) as f:
        summary = json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)

    embeds = data["embeds_2d"].astype(np.float32)  # (n, p, 2)
    attn = data["attn"].astype(np.float32)  # (n, batch, heads, seq, seq)
    fft_share = data["fft_share"].astype(np.float32)  # (n, p//2)
    share_at_kstar = data["share_at_kstar"].astype(np.float32).tolist()

    embeds.tofile(out_dir / "embeds.bin")
    attn.tofile(out_dir / "attn.bin")
    fft_share.tofile(out_dir / "fft_share.bin")

    meta = {
        "config": summary["config"],
        "n_frames": int(embeds.shape[0]),
        "p": int(embeds.shape[1]),
        "seq_len": int(attn.shape[-1]),
        "n_heads": int(attn.shape[2]),
        "fourier_k": int(data["fourier_k"]),
        "shapes": {
            "embeds": list(embeds.shape),
            "attn": list(attn.shape),
            "fft_share": list(fft_share.shape),
        },
        "share_at_kstar": share_at_kstar,
        "steps": data["steps"].astype(int).tolist(),
        "train_loss": data["train_loss"].astype(float).tolist(),
        "test_loss": data["test_loss"].astype(float).tolist(),
        "train_acc": data["train_acc"].astype(float).tolist(),
        "test_acc": data["test_acc"].astype(float).tolist(),
    }

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    size_bytes = sum(
        (out_dir / name).stat().st_size
        for name in [
            "embeds.bin",
            "attn.bin",
            "fft_share.bin",
            "meta.json",
        ]
    )
    print(f"wrote {out_dir} ({size_bytes / 1e6:.2f} MB total)")
    print(f"  meta.json:     {(out_dir / 'meta.json').stat().st_size / 1e3:.1f} KB")
    print(f"  embeds.bin:    {(out_dir / 'embeds.bin').stat().st_size / 1e3:.1f} KB ({embeds.shape})")
    print(f"  attn.bin:      {(out_dir / 'attn.bin').stat().st_size / 1e3:.1f} KB ({attn.shape})")
    print(f"  fft_share.bin: {(out_dir / 'fft_share.bin').stat().st_size / 1e3:.1f} KB ({fft_share.shape})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", default="checkpoints/raw/arrays_aligned.npz", type=Path)
    ap.add_argument("--summary", default="checkpoints/raw/summary.json", type=Path)
    ap.add_argument("--out", default="web/public/data", type=Path)
    args = ap.parse_args()
    export_web(args.arrays, args.summary, args.out)


if __name__ == "__main__":
    main()
