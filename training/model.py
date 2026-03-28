import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


def init_linear_(layer: nn.Linear, d: int) -> None:
    nn.init.normal_(layer.weight, std=1.0 / math.sqrt(d))


class AttentionHead(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.W_K = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.W_V = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.W_O = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)
        for m in (self.W_Q, self.W_K, self.W_V, self.W_O):
            init_linear_(m, cfg.d_model)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.seq_len, cfg.seq_len)).view(
                1, 1, cfg.seq_len, cfg.seq_len
            ),
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, T, _ = x.shape
        H, D = self.cfg.n_heads, self.cfg.d_head
        q = self.W_Q(x).view(B, T, H, D).transpose(1, 2)
        k = self.W_K(x).view(B, T, H, D).transpose(1, 2)
        v = self.W_V(x).view(B, T, H, D).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, H * D)
        out = self.W_O(out)
        if return_attn:
            return out, attn
        return out


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_mlp, bias=False)
        self.fc2 = nn.Linear(cfg.d_mlp, cfg.d_model, bias=False)
        init_linear_(self.fc1, cfg.d_model)
        init_linear_(self.fc2, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn = AttentionHead(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        attn_out = self.attn(x, return_attn=return_attn)
        if return_attn:
            attn_out, attn = attn_out
        x = x + attn_out
        x = x + self.mlp(x)
        if return_attn:
            return x, attn
        return x


class GrokTransformer(nn.Module):
    """Nanda et al. 2023 setup: 1-layer transformer for (a + b) mod p.

    No layer norm, no biases — matches the paper for reproducibility. Init
    scale 1/sqrt(d_model) across embeddings and linear layers. PyTorch's
    default embedding init (N(0, 1)) is ~11x too large here and prevents
    grokking from completing in 40k steps under wd=1.0.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.seq_len, cfg.d_model)
        nn.init.normal_(self.token_embed.weight, std=1.0 / math.sqrt(cfg.d_model))
        nn.init.normal_(self.pos_embed.weight, std=1.0 / math.sqrt(cfg.d_model))
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        nn.init.normal_(self.unembed.weight, std=1.0 / math.sqrt(cfg.vocab_size))

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.token_embed(x) + self.pos_embed(pos)
        attns = []
        for block in self.blocks:
            if return_attn:
                h, a = block(h, return_attn=True)
                attns.append(a)
            else:
                h = block(h)
        logits = self.unembed(h)
        if return_attn:
            return logits, attns
        return logits
