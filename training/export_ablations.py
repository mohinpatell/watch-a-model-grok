"""Copy ablation curves from checkpoints/ablations/ into the web bundle as
small per-run JSON files plus an index.

Run after `python -m training.ablate` completes."""

import argparse
import json
from pathlib import Path

from .ablate import ABLATIONS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="checkpoints/ablations", type=Path)
    ap.add_argument("--out", default="web/public/ablations", type=Path)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Only emit the canonical runs. Preserve their order (failures first,
    # then successes) as defined in ABLATIONS. Non-canonical extras stay in
    # checkpoints/ablations/ for reference but aren't exposed in the web UI.
    desired = [a["name"] for a in ABLATIONS if a.get("canonical")]
    available = {d.name for d in args.src.iterdir() if d.is_dir()}
    names: list[str] = [n for n in desired if n in available]

    written: list[str] = []
    for name in names:
        curve = args.src / name / "curve.json"
        if not curve.exists():
            continue
        with open(curve) as f:
            payload = json.load(f)
        with open(args.out / f"{name}.json", "w") as f:
            json.dump(payload, f)
        written.append(name)

    with open(args.out / "index.json", "w") as f:
        json.dump({"names": written}, f)

    print(f"wrote {len(written)} ablations to {args.out}: {written}")


if __name__ == "__main__":
    main()
