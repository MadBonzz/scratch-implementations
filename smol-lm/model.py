import torch
from torch import nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RMSNorm(nn.Module):
    def __init__(self, n_embed, eps=1e-5, device='cuda'):
        super(RMSNorm, self).__init__()
        self.gammas = nn.Parameter(torch.ones(n_embed)).to(device)
        self.eps = eps

    def forward(self, x):
        rms = torch.mean(torch.square(x), dim=1, dtype=x.dtype).unsqueeze(1) ** 0.5
        print(rms.shape)
        x = x / (rms + self.eps)
        return (x * self.gammas)
    
class RoPE(nn.Module):
    def __init__(self, n_embed, max_len, theta=10000, device='cuda'):
        super(RoPE, self).__init__()
        self.max_len = max_len
        angles = 1 / theta ** (torch.arange(0, n_embed, 2) / n_embed)
        sequences = torch.arange(max_len).float()
        theta_sequences = sequences.unsqueeze(1) @ angles.unsqueeze(0)
        vectors = torch.polar(torch.ones_like(theta_sequences), theta_sequences)
        self.register_buffer('rope', vectors.unsqueeze(0).to(device))
    
    def forward(self, x):
        B, T, C = x.shape
        assert T <= self.max_len
        complex_x = torch.view_as_complex(x.reshape(B, T, -1, 2))
        rotated_x = complex_x * self.rope[:, :T, :]
        return torch.view_as_real(rotated_x).reshape(B, T, C).to(x.device)
    
class GroupedQueryAttention(nn.Module):
    def __init__(self, n_heads, kv_heads, n_embed, head_dim, max_len, device='cuda'):
        super(GroupedQueryAttention, self).__init__()
        self.head_dim = head_dim
        self.repeat = n_heads // kv_heads
        self.q_proj = nn.Linear(n_embed, head_dim * n_heads)
        self.k_proj = nn.Linear(n_embed, head_dim * kv_heads)
        self.v_proj = nn.Linear(n_embed, head_dim * kv_heads)
        self.register_buffer('tril', torch.tril(torch.ones(max_len, max_len)))

    def attention(self, q, k, v):
        q = q.reshape(q.shape[:2], self.repeat, -1)


class MLP(nn.Module):
    def __init__(self, n_embed, hidden_size):
        super(MLP, self).__init()
        self.gate_proj = nn.Linear(n_embed, hidden_size)
        self.up_proj = nn.Linear(n_embed, hidden_size)
        self.down_proj = nn.Linear(hidden_size, n_embed)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(up * gate)


class SmolLM(nn.Module):
    def __init__(self, n_voacb, n_embed, n_layers, n_heads, kv_heads, head_dim, hidden_size, max_len, theta=10000, eps=1e-5, device='cuda'):
        super(SmolLM, self).__init__()
        self.max_len = max_len
        self.device = device
        self.embedding = nn.Embedding(n_voacb, n_embed)
        self.lm_head = nn.Linear(n_embed, n_voacb)

sample = torch.randn(4, 16, 32).to(device)


