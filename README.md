# watch a model grok

Training a 1-layer transformer to compute `(a + b) mod 113` and watching the internal representations during the memorize → plateau → grok transition.

After [Nanda et al. 2023, *Progress Measures for Grokking via Mechanistic Interpretability*](https://arxiv.org/abs/2301.05217).

Live: https://mohinpatell.github.io/watch-a-model-grok

## What you're looking at

A 1-layer transformer (4 heads, `d_model=128`) trained on 30% of the 113² possible `(a, b)` pairs, with AdamW and `weight_decay=1.0`. The run:

- memorizes the training set in ~140 steps (train acc → 1.0, test acc ≈ 0)
- plateaus for ~8000 steps (test loss climbs to a peak around step 1,500, then drifts down)
- snaps between steps 6,000 and 8,500 (test acc near-zero → 99% over ~2,000 steps)
- ends with a clean Fourier circuit — token embeddings sit on a ring at frequency `k = 46`

## Results

| step  | train acc | test acc | notes                             |
|-------|-----------|----------|-----------------------------------|
| 0     | 0.01      | 0.01     | chance                            |
| 494   | 1.00      | 0.04     | memorized, test at chance         |
| 3009  | 1.00      | 0.07     | plateau, nothing visible moving   |
| 5901  | 1.00      | 0.16     | onset, test loss peaks then falls |
| 7835  | 1.00      | 0.82     | grok — test acc snapping 0 → 1    |
| 40000 | 1.00      | 1.00     | generalized, Fourier ring stable  |

## What's actually happening

### Memorize → plateau → grok

The model sees 30% of the 113² possible `(a, b)` pairs, so it has two viable strategies:

1. **Memorize**: treat each of the ~3,800 training pairs as a lookup. Test accuracy stays at chance because there's no structure to generalize.
2. **Compute**: implement `(a + b) mod 113` as an actual algorithm. Works on any pair.

Memorization is available within a few hundred steps because the model has enough capacity to store every answer in weight space. Generalization isn't — it requires the weights to arrange themselves into a specific circuit that doesn't exist at init.

Weight decay (`wd = 1.0`) is what bridges them. A memorizing solution spreads large weights across many parameters; a generalizing solution concentrates weight norm on a few Fourier components. Weight decay keeps applying pressure toward lower norm, so once train loss is near zero and the gradient signal fades, the dominant force on the parameters is *shrink*. The model drifts through a plateau where both solutions coexist, then tips into the Fourier circuit because it's the lower-norm basin.

You can see this as a slow-motion phase transition: test accuracy sits at chance for thousands of steps, then snaps to 1.0 in a few thousand. The cliff edge is where the generalizing circuit becomes dominant enough to route the forward pass.

### The Fourier circuit

The generalized model computes `a + b mod 113` using a handful of Fourier modes. Token embeddings for the input symbols land on a 2D ring, spaced by angle `2π · k · x / 113` for a single frequency `k`. The attention + MLP then produce `cos(k·(a + b))` via the angle-sum identity:

```
cos(k·a) · cos(k·b) - sin(k·a) · sin(k·b) = cos(k·(a + b))
```

The unembedding subtracts each candidate `c` to form `cos(k·(a + b − c))`; summed across the active frequencies, this peaks sharply at `c ≡ a + b (mod p)`. You can see this in the viz: the scatter snaps onto a clean ring during the grok, and the attention heads specialize to route the two operands into the `=` position.

The specific frequencies (`{13, 14, 28, 31, 46}` for this seed, with `k = 46` slightly dominant) depend on the seed — any `k` coprime with 113 could in principle work. What's universal is the *shape* of the solution: circular embeddings, phase-based attention, a small number of active Fourier components in the final logits.

### Init scale matters

PyTorch's default `nn.Embedding` initializes with `N(0, 1)`, which for `d_model = 128` puts embeddings at distance `≈ √128 ≈ 11` from the origin. Weight decay has to drag them down over thousands of steps before its pressure meaningfully shapes the geometry — memorization happens in the meantime and the run stalls.

Scaling init by `1/√d_model` puts embeddings at radius `≈ 1` from step 0. From there, weight-decay pressure and the gradient signal are on the same order, so the Fourier solution is reachable without first burning thousands of steps undoing the initialization.

This is the single change that makes the run reliably grok inside 40k steps instead of wandering.

### Reading the viz

- **Loss (log x-axis)**: the memorize → plateau → grok shape is only legible in log-time. Train loss drops fast then hugs zero; test loss climbs, peaks, then cliffs.
- **Token embeddings (2D projection onto the `(cos, sin)` basis at `k = 46`, Procrustes-aligned)**: at init, a blob. After memorization, still a blob (the memorizing solution doesn't need geometry). During grok, the blob reorganizes into a ring. Procrustes alignment frame-to-frame means the ring stays still as weights rotate; without it the scatter would spin and you'd miss the structural change.
- **Attention routing**: the `=` token's attention weights over `a`, `b`, `=`, broken out per head. Pre-grok each head overshoots a one-operand preference; post-grok three of four heads settle near a balanced split.
- **FFT spectrum/spectrogram**: the share of embedding L2 power at each frequency. Early on it looks like noise; during grok, a few horizontal bands light up against uniform background.

## Repo layout

```
training/   PyTorch training loop, snapshot exporter, Fourier alignment
web/        Next.js 16 static site: scrub + scrollytelling viz
checkpoints/ training outputs (gitignored) + web-ready JSON bundles
```

## Training

```bash
uv sync
uv run python -m training.train
```

Runs on MPS or CPU. Checkpoints land in `checkpoints/raw/`. The pipeline is:

```bash
uv run python -m training.train           # writes arrays.npz + summary.json
uv run python -m training.align           # 2D project + Procrustes → arrays_aligned.npz
uv run python -m training.measures        # FFT power per checkpoint
uv run python -m training.export_web      # → web/public/data/*.bin + meta.json
```

The embedding projection is Procrustes-aligned frame-to-frame against the Fourier (cos, sin) basis at the dominant frequency of the final embedding (k = 46 for `seed=42`) so the ring stays still as weights rotate.

### Ablations

```bash
uv run python -m training.ablate --num-steps 15000  # ~3 min per run on MPS
uv run python -m training.export_ablations          # → web/public/ablations/*.json
```

Sweeps `weight_decay`, `init_mode`, `use_layer_norm`, and a couple of alternative seeds. Each run writes a compact curve of train/test loss and accuracy at ~70 log-spaced checkpoints.

## Web

```bash
cd web
pnpm install
pnpm dev
```

Production build outputs static HTML:

```bash
cd web && pnpm build    # emits web/out/
```

Stack: Next.js 16 (static export), TypeScript, Tailwind v4, Zustand for client state, Observable Plot for line charts, Canvas 2D for the ring scatter, KaTeX for equations.

## Deploy

GitHub Pages via `.github/workflows/deploy.yml`. Pushes to `main` build `web/` and publish `web/out/` — no CI secrets required. Next.js is configured to apply a `/watch-a-model-grok` basePath only when `GITHUB_PAGES=true` (set by the workflow), so local `pnpm dev` still serves at `/`.
