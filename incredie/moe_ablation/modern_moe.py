"""modern_moe.py — Mixture of Experts (DeepSeek V4, Slides 15-22).

Replaces the dense MLP with a DeepSeekMoE module:
  - 32 SwiGLU experts (fine-grained: intermediate_dim = hidden // 2)
  - 1 shared SwiGLU expert (always active)
  - Independent scoring router (768 -> 32, sqrt_softplus activation)
  - top-4 selection by score + dynamic load-balancing bias
  - Aux-loss-free load balancing (per-expert bias, updated each microbatch)
  - Hash-table routing for shallow layers (first 3)
  - Routed scaling ×1.5 after within-4 normalization

All other modern features (RoPE, RMSNorm, no bias, QK-norm) are kept.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model.gpt import GPTConfig, apply_rotary_emb, norm, CausalSelfAttention


# ---- activation ----
def sqrt_softplus(x):
    """Non-negative, monotonic activation. Used for independent expert scoring."""
    return torch.sqrt(F.softplus(x))


# ---- single SwiGLU expert ----
class SwiGLUExpert(nn.Module):
    """Fine-grained SwiGLU expert: intermediate dim < hidden dim."""

    def __init__(self, n_embd, intermediate_dim):
        super().__init__()
        self.gate = nn.Linear(n_embd, intermediate_dim, bias=False)
        self.up = nn.Linear(n_embd, intermediate_dim, bias=False)
        self.down = nn.Linear(intermediate_dim, n_embd, bias=False)

    def forward(self, x):
        h = F.silu(self.gate(x)) * self.up(x)
        return self.down(h)


# ---- MoE router ----
class MoERouter(nn.Module):
    """Independent scoring router with dynamic load-balancing bias."""

    def __init__(self, n_embd, n_experts, top_k=4):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.score_proj = nn.Linear(n_embd, n_experts, bias=False)
        # Dynamic bias for load balancing (NOT a parameter — no gradient)
        self.register_buffer("bias", torch.zeros(n_experts))

    def forward(self, x, hash_indices=None):
        """
        Args:
            x: (N, C) — flattened token representations
            hash_indices: (N, top_k) — optional hash-table indices for shallow layers
        Returns:
            expert_indices: (N, top_k) — which experts each token routes to
            weights: (N, top_k) — normalized weights for the selected experts
        """
        N = x.shape[0]

        # Independent scores (no softmax across experts)
        logits = self.score_proj(x)                     # (N, n_experts)
        scores = sqrt_softplus(logits)                   # independent, non-negative

        # Add load-balancing bias (detached from gradient)
        biased_scores = scores + self.bias.detach()      # (N, n_experts)

        # Top-k selection
        if hash_indices is not None:
            # Hash-table routing: expert selection is fixed, scores still used for weights
            expert_indices = hash_indices                # (N, top_k)
        else:
            _, expert_indices = torch.topk(biased_scores, self.top_k, dim=-1)  # (N, top_k)

        # Gather selected scores and normalize within the k
        selected_scores = torch.gather(scores, 1, expert_indices)  # (N, top_k)
        weights = selected_scores / selected_scores.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        weights = weights * 1.5  # routed_scaling (DeepSeek constant)

        return expert_indices, weights, scores


# ---- MoE module (replaces MLP) ----
class DeepSeekMoE(nn.Module):
    """32 routed SwiGLU experts + 1 shared expert, with load-balanced routing."""

    def __init__(self, config, n_experts=32, top_k=4, use_hash_routing=False, vocab_size=50304):
        super().__init__()
        n_embd = config.n_embd
        self.n_experts = n_experts
        self.top_k = top_k
        self.n_embd = n_embd
        self.use_hash_routing = use_hash_routing

        # Fine-grained experts: intermediate_dim = n_embd // 2 (DeepSeekMoE style)
        expert_dim = n_embd // 2
        self.experts = nn.ModuleList([SwiGLUExpert(n_embd, expert_dim) for _ in range(n_experts)])

        # Shared expert (always active, weight=1.0)
        self.shared_expert = SwiGLUExpert(n_embd, n_embd // 2)

        # Router
        self.router = MoERouter(n_embd, n_experts, top_k)

        # Frozen hash table for shallow-layer routing
        if use_hash_routing:
            self.register_buffer("hash_table", self._make_hash_table(vocab_size, n_experts, top_k))

    def _make_hash_table(self, vocab_size, n_experts, top_k):
        """Create a balanced random hash table: each expert appears equally often."""
        table = torch.zeros(vocab_size, top_k, dtype=torch.long)
        slots_per_expert = vocab_size * top_k // n_experts
        # Fill with balanced assignments
        indices = torch.arange(vocab_size * top_k) % n_experts
        indices = indices[torch.randperm(vocab_size * top_k)]
        table = indices.reshape(vocab_size, top_k)
        return table

    def update_load_balance(self, expert_indices, n_tokens):
        """Aux-loss-free load balancing: adjust per-expert bias.
        Called after each microbatch step (no gradient)."""
        with torch.no_grad():
            # Count how many tokens each expert received
            counts = torch.bincount(expert_indices.flatten(), minlength=self.n_experts)
            expected = self.top_k * n_tokens / self.n_experts
            # busy experts: bias down; idle experts: bias up
            adjustment = torch.where(
                counts > expected,
                torch.tensor(-0.001, device=counts.device),
                torch.tensor(0.001, device=counts.device)
            )
            self.router.bias.add_(adjustment)

    def forward(self, x, token_ids=None):
        """
        Args:
            x: (B, T, C)
            token_ids: (B, T) — token ids, needed for hash-table routing in shallow layers
        Returns:
            out: (B, T, C)
        """
        B, T, C = x.shape
        N = B * T
        flat_x = x.reshape(N, C)

        # Hash-table routing for shallow layers
        hash_indices = None
        if self.use_hash_routing and token_ids is not None:
            flat_ids = token_ids.reshape(-1)
            hash_indices = self.hash_table[flat_ids]  # (N, top_k)

        # Route
        expert_indices, weights, scores = self.router(flat_x, hash_indices)  # (N, top_k), (N, top_k)

        # Initialize output
        out = torch.zeros(N, C, device=x.device, dtype=x.dtype)

        # Compute expert outputs (efficient batching per expert)
        for e_idx in range(self.n_experts):
            # Find (token_index, rank) pairs where this expert is selected
            token_rows, rank_cols = torch.where(expert_indices == e_idx)
            n_selected = token_rows.numel()
            if n_selected == 0:
                continue

            # Gather tokens and their weights
            selected_x = flat_x[token_rows]                   # (n_selected, C)
            selected_w = weights[token_rows, rank_cols]       # (n_selected,)

            # Compute expert output and scatter-add
            expert_out = self.experts[e_idx](selected_x)      # (n_selected, C)
            out.index_add_(0, token_rows, expert_out * selected_w.unsqueeze(-1))

        # Add shared expert
        out = out + self.shared_expert(flat_x)

        # Reshape and update load balancing
        out = out.reshape(B, T, C)
        self.update_load_balance(expert_indices, N)

        return out


# ---- Block with MoE ----
class MoE_Block(nn.Module):
    def __init__(self, config, layer_idx, n_experts=8, top_k=1):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        # First 3 layers use hash-table routing
        use_hash = layer_idx < 3
        self.mlp = DeepSeekMoE(config, n_experts=n_experts, top_k=top_k,
                               use_hash_routing=use_hash,
                               vocab_size=config.vocab_size)
        self.layer_idx = layer_idx

    def forward(self, x, cos_sin, kv_cache=None, block_mask=None, token_ids=None):
        x = x + self.attn(norm(x), cos_sin, kv_cache, block_mask)
        x = x + self.mlp(norm(x), token_ids=token_ids)
        return x


# ---- Trunk ----
class ModernMoETrunk(nn.Module):
    """Modern GPT trunk with DeepSeekMoE replacing dense MLP."""

    Config = GPTConfig

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_experts = 8
        self.top_k = 1

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([
                MoE_Block(config, i, n_experts=self.n_experts, top_k=self.top_k)
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
            # Zero-init all expert down projections (standard DeepSeekMoE practice)
            for expert in block.mlp.experts:
                torch.nn.init.zeros_(expert.down.weight)
            torch.nn.init.zeros_(block.mlp.shared_expert.down.weight)
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
        # Rough estimate: same for now
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
            x = block(x, cos_sin, kv_cache, block_mask, token_ids=idx)
        x = norm(x)
        return x
