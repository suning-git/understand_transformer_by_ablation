"""modern_swa.py — Sliding Window Attention + Sink (DeepSeek V4, Slides 23, 28).

Single-delta optimization on top of the modern core GPT:
  - Replace full causal attention with sliding window (window_size=96).
    Each query attends only to the most recent window_size tokens (including itself).
  - Add per-head sink logit: a learnable scalar that gives each head a "dump nothing"
    outlet. After softmax the sink mass is discarded (no value to aggregate).

All other modern features (RoPE, RMSNorm, no bias, ReLU^2, QK-norm) are kept.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model.gpt import GPTConfig, apply_rotary_emb, norm


# ---- MLP unchanged ----
from core.model.gpt import MLP


class SWA_CausalSelfAttention(nn.Module):
    """Sliding window causal attention with per-head sink token.

    window_size: max number of past tokens each query can attend to.
    sink: a learnable per-head scalar logit appended before softmax.
    """

    def __init__(self, config, layer_idx, window_size=96):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.window_size = window_size
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.enable_gqa = self.n_kv_head != self.n_head

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # Sink: one learnable logit per head (initialized to 0, learns to be negative)
        self.sink_logit = nn.Parameter(torch.zeros(self.n_head))

    def _make_sliding_window_mask(self, T, device, dtype):
        """Create banded causal mask: (T, T) with window_size bandwidth.
        mask[t, s] = True (keep) if s <= t and s > t - window_size.
        """
        # Causal mask
        causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        # Sliding window: only keep within window_size
        row_indices = torch.arange(T, device=device).unsqueeze(1)
        col_indices = torch.arange(T, device=device).unsqueeze(0)
        window = (col_indices > row_indices - self.window_size)
        return causal & window  # (T, T)

    def forward(self, x, cos_sin, kv_cache, block_mask=None):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2)
        Tk = k.size(2)

        # Compute attention logits
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, n_head, Tq, Tk)

        # Sliding window mask
        sw_mask = self._make_sliding_window_mask(Tk, attn_logits.device, attn_logits.dtype)
        # Broadcast mask: (1, 1, Tq, Tk) — same mask for all heads/batch
        sw_mask = sw_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, Tk, Tk)
        sw_mask = sw_mask[:, :, :Tq, :]              # (1, 1, Tq, Tk)
        attn_logits = attn_logits.masked_fill(~sw_mask, float("-inf"))

        # Sink: append per-head logit. Expand to (B, n_head, Tq, 1)
        sink = self.sink_logit.view(1, self.n_head, 1, 1).expand(B, -1, Tq, 1)
        attn_logits = torch.cat([attn_logits, sink], dim=-1)  # (B, n_head, Tq, Tk+1)

        # Softmax (sink included in denominator)
        attn_weights = F.softmax(attn_logits, dim=-1)

        # Discard sink weight (last column), aggregate values
        v_weighted = attn_weights[:, :, :, :Tk] @ v  # (B, n_head, Tq, head_dim)

        y = v_weighted.transpose(1, 2).contiguous().view(B, Tq, C)
        y = self.c_proj(y)
        return y


class SWA_Block(nn.Module):
    def __init__(self, config, layer_idx, window_size=96):
        super().__init__()
        self.attn = SWA_CausalSelfAttention(config, layer_idx, window_size=window_size)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache=None, block_mask=None):
        x = x + self.attn(norm(x), cos_sin, kv_cache, block_mask)
        x = x + self.mlp(norm(x))
        return x


class ModernSWATrunk(nn.Module):
    """Modern GPT trunk with Sliding Window Attention + Sink."""

    Config = GPTConfig

    def __init__(self, config, window_size=96):
        super().__init__()
        self.config = config
        self.window_size = window_size
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([
                SWA_Block(config, i, window_size=window_size)
                for i in range(config.n_layer)
            ]),
        })
        self.type_emb = nn.Embedding(config.n_token_types, config.n_embd)

        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        torch.nn.init.zeros_(self.type_emb.weight)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            self.type_emb.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    @property
    def blocks(self):
        return self.transformer.h

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h = self.config.n_layer, self.config.n_head
        q, t = self.config.n_embd // self.config.n_head, self.config.sequence_len
        return 6 * (nparams - nparams_embedding) + 12 * l * h * q * t

    def get_device(self):
        return self.transformer.wte.weight.device

    def forward(self, idx, token_types=None, kv_cache=None, block_mask=None):
        B, T = idx.shape
        device = idx.device
        x = self.transformer.wte(idx)
        if token_types is not None:
            x = x + self.type_emb(token_types)
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache, block_mask)
        x = norm(x)
        return x
