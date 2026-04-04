"""Ablation sweep — vary a handful of knobs and record train/test curves so
we can show what does and doesn't grok. Cheap per-run: no embedding or attn
snapshots, just scalar curves at log-spaced checkpoints."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import DEFAULT, Config
from .data import make_dataset, train_test_split
from .model import GrokTransformer


ABLATIONS: list[dict] = [
    # The baseline is the published run; we do NOT retrain it here. The
    # variants below each change one thing relative to it. The six entries
    # tagged canonical=True are what the web UI shows; everything else is
    # an extra robustness check.
    {"name": "wd_low", "weight_decay": 0.01, "canonical": True},
    {"name": "wd_zero", "weight_decay": 0.0, "canonical": True},
    {"name": "init_pytorch", "init_mode": "pytorch", "canonical": True},
    {"name": "layer_norm", "use_layer_norm": True, "canonical": True},
    {"name": "seed_0", "seed": 0, "canonical": True},
    {"name": "seed_7", "seed": 7, "canonical": True},
    # Two extra LayerNorm runs across seeds to check whether the ~3k-step
    # grok observed on seed=42 is seed-robust or a fluke. Not shown in the
    # web table — referenced from the essay in prose.
    {"name": "layer_norm_s0", "use_layer_norm": True, "seed": 0},
    {"name": "layer_norm_s7", "use_layer_norm": True, "seed": 7},
]


def curve_schedule(num_steps: int, num_points: int) -> list[int]:
    s = np.unique(
        np.round(np.logspace(0, np.log10(num_steps), num_points)).astype(int)
    )
    s = np.concatenate([[0], s, [num_steps]])
    s = np.unique(np.clip(s, 0, num_steps))
    return s.tolist()


def run_one(name: str, cfg: Config, out_dir: Path, device: str) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    inputs, targets = make_dataset(cfg, device=device)
    train_x, train_y, test_x, test_y = train_test_split(inputs, targets, cfg)

    model = GrokTransformer(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
    )

    ckpts = set(curve_schedule(cfg.num_steps, num_points=80))
    records: list[dict] = []

    pbar = tqdm(range(cfg.num_steps + 1), desc=f"ablate:{name}", smoothing=0.01)
    for step in pbar:
        if step in ckpts:
            model.train(False)
            with torch.no_grad():
                tr = model(train_x)[:, -1, :]
                te = model(test_x)[:, -1, :]
                rec = {
                    "step": step,
                    "train_loss": F.cross_entropy(tr, train_y).item(),
                    "test_loss": F.cross_entropy(te, test_y).item(),
                    "train_acc": (tr.argmax(-1) == train_y).float().mean().item(),
                    "test_acc": (te.argmax(-1) == test_y).float().mean().item(),
                }
            model.train(True)
            records.append(rec)
            pbar.set_postfix(
                tr=f"{rec['train_loss']:.3f}",
                te=f"{rec['test_loss']:.3f}",
                acc=f"{rec['test_acc']:.2f}",
            )

        if step == cfg.num_steps:
            break

        logits = model(train_x)
        loss = F.cross_entropy(logits[:, -1, :], train_y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "config": {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")},
        "records": records,
        "final": records[-1],
    }
    with open(run_dir / "curve.json", "w") as f:
        json.dump(payload, f)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoints/ablations", type=Path)
    ap.add_argument("--num-steps", type=int, default=None,
                    help="override steps for quicker testing")
    ap.add_argument("--only", default=None,
                    help="comma-separated ablation names to run")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"device: {device}")

    args.out.mkdir(parents=True, exist_ok=True)

    only = set(args.only.split(",")) if args.only else None

    summary: list[dict] = []
    for ab in ABLATIONS:
        if only and ab["name"] not in only:
            continue
        overrides = {k: v for k, v in ab.items() if k not in {"name", "canonical"}}
        if args.num_steps is not None:
            overrides["num_steps"] = args.num_steps
        cfg = replace(DEFAULT, **overrides)
        print(f"\n=== {ab['name']} ===")
        print(f"  overrides: {overrides}")
        payload = run_one(ab["name"], cfg, args.out, device)
        final = payload["final"]
        print(
            f"  final: train_acc={final['train_acc']:.3f} "
            f"test_acc={final['test_acc']:.3f}"
        )
        summary.append({
            "name": ab["name"],
            "overrides": overrides,
            "final": final,
        })

    with open(args.out / "summary.json", "w") as f:
        json.dump({"ablations": summary}, f, indent=2)
    print(f"\nwrote summary to {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
