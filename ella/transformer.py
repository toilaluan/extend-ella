import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_default_rope_parameters(
    head_dim: int,
    base: int = 10000,
    device = "cuda"
) -> tuple["torch.Tensor", float]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / head_dim))
    return inv_freq

class RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, head_dim: int, device: str = "cuda"):
        super().__init__()
        inv_freq = _compute_default_rope_parameters(head_dim=head_dim, base=10000, device=device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device) # 1, d//2, 1
        position_ids_expanded = position_ids[:, None, :].float() # B, 1, L

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2) # B, L, d//2
            emb = torch.cat((freqs, freqs), dim=-1) # B, L, d
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # cos, sin: B, L, D
    # q, k: B, H, L, D
    # unsqueeze_dim is usually equal 1 to broadcast head-dimension
    sin = sin.unsqueeze(unsqueeze_dim)
    cos = cos.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int = 8, head_dim: int = 64):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * head_dim
        self.head_dim = head_dim
        self.norm_q = torch.nn.RMSNorm(head_dim, eps=1e-6, elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(head_dim, eps=1e-6, elementwise_affine=True)
        self.to_q = nn.Linear(hidden_size, self.inner_dim)
        self.to_k = nn.Linear(hidden_size, self.inner_dim)
        self.to_v = nn.Linear(hidden_size, self.inner_dim)
        self.to_out = nn.Linear(self.inner_dim, hidden_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, position_embeddings = None):
        B, L, D = hidden_states.shape
        hidden_shape = (B, L, -1, self.head_dim)
        q = self.to_q(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.to_k(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.to_v(hidden_states).view(hidden_shape).transpose(1, 2)

        q = self.norm_q(q)
        k = self.norm_k(k)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)

        out = self.to_out(out)

        return out

class Block(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: int = 4, num_attention_heads: int = 8, head_dim: int = 64):
        super().__init__()
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.pre_norm = nn.RMSNorm(hidden_size)
        self.post_norm = nn.RMSNorm(hidden_size)
        self.proj_mlp = nn.Linear(hidden_size, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(self.mlp_hidden_dim, hidden_size)

        self.attn = Attention(hidden_size=hidden_size, num_attention_heads=num_attention_heads, head_dim=head_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None, position_embeddings = None):
        residual = hidden_states
        norm_hidden_states = self.pre_norm(hidden_states)
        hidden_states = self.attn(norm_hidden_states, attention_mask, position_embeddings)
        
        hidden_states = hidden_states + residual

        residual = hidden_states

        hidden_states = self.post_norm(hidden_states)

        hidden_states = self.proj_out(self.act_mlp(self.proj_mlp(hidden_states)))

        hidden_states = residual + hidden_states

        return hidden_states

if __name__ == "__main__":
    block = Block(512, 2, 8, 32)
    rotary = RotaryEmbedding(device="cpu", head_dim=32)

    position_ids = torch.arange(0, 128)[None, :]


    x = torch.zeros((1, 128, 512))
    pos_emb = rotary(x, position_ids)

    print(pos_emb[0])

    out = block(x, position_embeddings=pos_emb)

    print(out.shape)