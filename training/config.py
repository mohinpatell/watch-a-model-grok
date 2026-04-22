from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Task: (a + b) mod p, tokens 0..p-1 plus "=" as separator.
    p: int = 113
    frac_train: float = 0.3

    # Nanda et al. 2023 architecture.
    n_layers: int = 1
    n_heads: int = 4
    d_model: int = 128
    d_head: int = 32
    d_mlp: int = 512

    # Training.
    lr: float = 1e-3
    weight_decay: float = 1.0
    betas: tuple[float, float] = (0.9, 0.98)
    num_steps: int = 40_000
    batch_size: int = 512
    seed: int = 42

    # Checkpointing. Log-spaced to capture both memorization and grok phases.
    num_checkpoints: int = 300

    # Ablation knobs. Defaults match the main paper setup.
    init_mode: str = "sqrt_d"   # "sqrt_d" | "pytorch"; embeddings + linears scale.
    use_layer_norm: bool = False  # add pre-norm LayerNorm blocks.

    @property
    def vocab_size(self) -> int:
        # p number tokens plus one separator token ("=").
        return self.p + 1

    @property
    def eq_token(self) -> int:
        return self.p

    @property
    def seq_len(self) -> int:
        # [a, b, =], predicting the sum at the "=" position.
        return 3


DEFAULT = Config()
