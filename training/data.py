import torch

from .config import Config


def make_dataset(cfg: Config, device: str = "cpu"):
    """All (a, b) pairs for a, b in [0, p). Each example is [a, b, eq] with
    target (a + b) mod p predicted at the final position."""
    p = cfg.p
    a = torch.arange(p, device=device).repeat_interleave(p)
    b = torch.arange(p, device=device).repeat(p)
    eq = torch.full_like(a, cfg.eq_token)
    inputs = torch.stack([a, b, eq], dim=-1)
    targets = (a + b) % p
    return inputs, targets


def train_test_split(inputs: torch.Tensor, targets: torch.Tensor, cfg: Config):
    g = torch.Generator(device=inputs.device).manual_seed(cfg.seed)
    n = inputs.shape[0]
    perm = torch.randperm(n, generator=g, device=inputs.device)
    n_train = int(n * cfg.frac_train)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    return (
        inputs[train_idx],
        targets[train_idx],
        inputs[test_idx],
        targets[test_idx],
    )
