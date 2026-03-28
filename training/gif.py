"""Render a matplotlib GIF of the training run: loss curves + embedding ring
+ attention heatmap, synchronized to step. This is the kill-or-proceed check
before committing to the web build — if this GIF isn't visually striking,
stop here."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter, FuncAnimation


def render_gif(arrays_path: Path, summary_path: Path, out_path: Path, fps: int = 12) -> None:
    data = np.load(arrays_path)
    with open(summary_path) as f:
        summary = json.load(f)

    steps = data["steps"]
    train_loss = data["train_loss"]
    test_loss = data["test_loss"]
    embeds_2d = data["embeds_2d"]  # (n, p, 2)
    attn = data["attn"]  # (n, batch, heads, seq, seq)
    n_frames = len(steps)
    p = embeds_2d.shape[1]

    # Fixed axis limits across the whole animation — computed from all frames.
    pad = 0.1
    all_xy = embeds_2d.reshape(-1, 2)
    xmin, ymin = all_xy.min(axis=0) - pad
    xmax, ymax = all_xy.max(axis=0) + pad
    span = max(xmax - xmin, ymax - ymin)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = span / 2
    emb_xlim = (cx - half, cx + half)
    emb_ylim = (cy - half, cy + half)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=100)
    plt.subplots_adjust(wspace=0.25, left=0.05, right=0.98, top=0.88, bottom=0.12)

    # Loss curve axis.
    ax_loss = axes[0]
    ax_loss.set_xscale("log")
    ax_loss.set_yscale("log")
    ax_loss.set_xlim(1, max(steps[-1], 2))
    ax_loss.set_ylim(min(min(train_loss), min(test_loss)) * 0.5, max(max(train_loss), max(test_loss)) * 2)
    ax_loss.set_xlabel("training step")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_title("loss")
    ax_loss.grid(True, which="both", alpha=0.2)
    line_train, = ax_loss.plot([], [], color="#2563eb", lw=2, label="train")
    line_test, = ax_loss.plot([], [], color="#dc2626", lw=2, label="test")
    ax_loss.legend(loc="upper right")
    vline = ax_loss.axvline(1, color="#333", lw=1, alpha=0.6)

    # Embedding ring axis.
    ax_emb = axes[1]
    ax_emb.set_xlim(*emb_xlim)
    ax_emb.set_ylim(*emb_ylim)
    ax_emb.set_aspect("equal")
    ax_emb.set_xticks([])
    ax_emb.set_yticks([])
    ax_emb.set_title("token embeddings (PCA 2D)")
    scatter = ax_emb.scatter(
        np.zeros(p), np.zeros(p),
        c=np.arange(p), cmap="twilight", s=30, edgecolor="white", linewidth=0.3
    )

    # Attention heatmap — average over batch and heads, layer 0.
    ax_attn = axes[2]
    ax_attn.set_title("attention (head 0, probe pair 0)")
    ax_attn.set_xticks([0, 1, 2], ["a", "b", "="])
    ax_attn.set_yticks([0, 1, 2], ["a", "b", "="])
    im = ax_attn.imshow(
        np.zeros((3, 3)), cmap="magma", vmin=0, vmax=1, aspect="equal"
    )
    plt.colorbar(im, ax=ax_attn, fraction=0.046, pad=0.04)

    title = fig.suptitle("", fontsize=14)

    def update(i: int):
        step = int(steps[i])
        line_train.set_data(steps[: i + 1], train_loss[: i + 1])
        line_test.set_data(steps[: i + 1], test_loss[: i + 1])
        vline.set_xdata([max(step, 1), max(step, 1)])

        scatter.set_offsets(embeds_2d[i])

        a = attn[i, 0, 0]  # first probe pair, first head
        im.set_data(a)

        test_acc = data["test_acc"][i]
        title.set_text(
            f"step {step:>6}   |   test acc {test_acc:.2f}   "
            f"|   train {train_loss[i]:.3f}   test {test_loss[i]:.3f}"
        )
        return line_train, line_test, vline, scatter, im, title

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)
    writer = PillowWriter(fps=fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, {n_frames} frames)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", default="checkpoints/raw/arrays_aligned.npz", type=Path)
    ap.add_argument("--summary", default="checkpoints/raw/summary.json", type=Path)
    ap.add_argument("--out", default="checkpoints/raw/training.gif", type=Path)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()
    render_gif(args.arrays, args.summary, args.out, fps=args.fps)


if __name__ == "__main__":
    main()
