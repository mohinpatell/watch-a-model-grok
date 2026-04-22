"""Train a 1-layer transformer on (a + b) mod p and dump checkpoints suitable
for scroll-driven visualization. Canonical Nanda et al. 2023 setup."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import DEFAULT, Config
from .data import make_dataset, train_test_split
from .model import GrokTransformer


def checkpoint_schedule(num_steps: int, num_checkpoints: int) -> list[int]:
    """Log-spaced steps, so we capture the rapid memorization phase, the long
    plateau, and the grok snap without wasting checkpoints on flat regions."""
    steps = np.unique(
        np.round(np.logspace(0, np.log10(num_steps), num_checkpoints)).astype(int)
    )
    steps = np.concatenate([[0], steps])
    steps = np.unique(np.clip(steps, 0, num_steps))
    return steps.tolist()


@torch.no_grad()
def capture_checkpoint(
    model: GrokTransformer,
    step: int,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    attn_probe_x: torch.Tensor,
) -> dict:
    """Capture everything needed for one frame of the visualization."""
    was_training = model.training
    model.train(False)

    train_logits = model(train_x)
    test_logits = model(test_x)
    train_loss = F.cross_entropy(train_logits[:, -1, :], train_y).item()
    test_loss = F.cross_entropy(test_logits[:, -1, :], test_y).item()
    train_acc = (train_logits[:, -1, :].argmax(-1) == train_y).float().mean().item()
    test_acc = (test_logits[:, -1, :].argmax(-1) == test_y).float().mean().item()

    p = model.cfg.p
    embeds = model.token_embed.weight[:p].detach().cpu().numpy().astype(np.float32)

    _, attns = model(attn_probe_x, return_attn=True)
    attn = attns[0].detach().cpu().numpy().astype(np.float32)

    model.train(was_training)
    return {
        "step": step,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "embeds": embeds,
        "attn": attn,
    }


def save_snapshots(snapshots: list[dict], cfg: Config, out_dir: Path) -> None:
    """Persist snapshots as .npz arrays + a JSON summary, no pickle."""
    n = len(snapshots)
    steps = np.array([s["step"] for s in snapshots], dtype=np.int64)
    train_loss = np.array([s["train_loss"] for s in snapshots], dtype=np.float32)
    test_loss = np.array([s["test_loss"] for s in snapshots], dtype=np.float32)
    train_acc = np.array([s["train_acc"] for s in snapshots], dtype=np.float32)
    test_acc = np.array([s["test_acc"] for s in snapshots], dtype=np.float32)
    embeds = np.stack([s["embeds"] for s in snapshots], axis=0)
    attn = np.stack([s["attn"] for s in snapshots], axis=0)

    np.savez_compressed(
        out_dir / "arrays.npz",
        steps=steps,
        train_loss=train_loss,
        test_loss=test_loss,
        train_acc=train_acc,
        test_acc=test_acc,
        embeds=embeds,
        attn=attn,
    )

    summary = {
        "config": cfg.__dict__,
        "num_snapshots": n,
        "steps": steps.tolist(),
        "train_loss": train_loss.tolist(),
        "test_loss": test_loss.tolist(),
        "train_acc": train_acc.tolist(),
        "test_acc": test_acc.tolist(),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f)


def train(cfg: Config, out_dir: Path, device: str = "cpu") -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    out_dir.mkdir(parents=True, exist_ok=True)

    inputs, targets = make_dataset(cfg, device=device)
    train_x, train_y, test_x, test_y = train_test_split(inputs, targets, cfg)

    probe_pairs = [(0, 0), (5, 10), (30, 40), (70, 43), (100, 100)]
    attn_probe_x = torch.tensor(
        [[a, b, cfg.eq_token] for a, b in probe_pairs], device=device
    )

    print(
        f"dataset: {inputs.shape[0]} total pairs, "
        f"{train_x.shape[0]} train, {test_x.shape[0]} test"
    )

    model = GrokTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} parameters")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
    )

    checkpoints_at = set(checkpoint_schedule(cfg.num_steps, cfg.num_checkpoints))
    snapshots: list[dict] = []

    pbar = tqdm(range(cfg.num_steps + 1), desc="train", smoothing=0.01)
    for step in pbar:
        if step in checkpoints_at:
            snap = capture_checkpoint(
                model, step, train_x, train_y, test_x, test_y, attn_probe_x
            )
            snapshots.append(snap)
            pbar.set_postfix(
                train=f"{snap['train_loss']:.3f}",
                test=f"{snap['test_loss']:.3f}",
                acc=f"{snap['test_acc']:.2f}",
            )

        if step == cfg.num_steps:
            break

        logits = model(train_x)
        loss = F.cross_entropy(logits[:, -1, :], train_y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    save_snapshots(snapshots, cfg, out_dir)
    print(f"saved {len(snapshots)} snapshots to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="checkpoints/raw", type=Path)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = DEFAULT
    if args.num_steps is not None:
        cfg = Config(**{**cfg.__dict__, "num_steps": args.num_steps})

    device = args.device or (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"device: {device}")
    train(cfg, args.out, device=device)


if __name__ == "__main__":
    main()
