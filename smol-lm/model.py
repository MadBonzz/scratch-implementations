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
    def __init__(self, n_heads, kv_heads, n_embed, head_dim, max_len, rope, device='cuda'):
        super(GroupedQueryAttention, self).__init__()
        self.max_len = max_len
        self.head_dim = head_dim
        self.kv_heads = kv_heads
        self.repeat = n_heads // kv_heads
        self.q_proj = nn.Linear(n_embed, head_dim * n_heads)
        self.k_proj = nn.Linear(n_embed, head_dim * kv_heads)
        self.v_proj = nn.Linear(n_embed, head_dim * kv_heads)
        self.rope = rope

    def attention(self, q, k, v):
        B, T, C = q.shape
        k = k.repeat(1, 1, 3)
        v = v.repeat(1, 1, 3)
        # q = q.reshape(B, T, -1, self.head_dim)
        # v = v.unsqueeze(2)
        # k = k.unsqueeze(2)
        # k = k.expand(k.shape[:2], self.repeat, self.head_dim)
        # v = v.expand(v.shape[:2], self.repeat, self.head_dim)
        q = self.rope(q)
        k = self.rope(k)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return attn_out
    
    def forward(self, x):
        B, T, C = x.shape
        assert T < self.max_len
        q_vec = self.q_proj(x)
        k_vec = self.k_proj(x)
        v_vec = self.v_proj(x)
        out = torch.empty(x.shape)
        for i in range(self.kv_heads):
            out[:, :, i*self.repeat*self.head_dim:(i+1)*self.repeat*self.head_dim] = self.attention(q_vec[:, :, i*self.repeat*self.head_dim:(i+1)*self.repeat*self.head_dim],
                                                                                                    k_vec[:, :, i*self.head_dim:(i+1)*self.head_dim],
                                                                                                    v_vec[:, :, i*self.head_dim:(i+1)*self.head_dim]) 
        return out


class MLP(nn.Module):
    def __init__(self, n_embed, hidden_size):
        super(MLP, self).__init__()
        self.gate_proj = nn.Linear(n_embed, hidden_size)
        self.up_proj = nn.Linear(n_embed, hidden_size)
        self.down_proj = nn.Linear(hidden_size, n_embed)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(self.act_fn(up * gate))


class Decoder(nn.Module):
    def __init__(self, n_heads, kv_heads, n_embed, head_dim, hidden_size, max_len, rope, eps=1e-5, device='cuda'):
        super(Decoder, self).__init__()
        self.input_norm = RMSNorm(n_embed, eps, device)
        self.out_norm = RMSNorm(n_embed, eps, device)
        self.attn = GroupedQueryAttention(n_heads, kv_heads, n_embed, head_dim, max_len, rope, device)
        self.mlp = MLP(n_embed, hidden_size)

    def forward(self, x):
        x = self.input_norm(x)
        x = x + self.attn(x).to(device)
        x = self.out_norm(x)
        x = x + self.mlp(x)
        return x

class SmolLM(nn.Module):
    def __init__(self, n_voacb, n_embed, n_layers, n_heads, kv_heads, head_dim, hidden_size, max_len, theta=10000, eps=1e-5, device='cuda'):
        super(SmolLM, self).__init__()
        self.max_len = max_len
        self.device = device
        self.embedding = nn.Embedding(n_voacb, n_embed)
        self.lm_head = nn.Linear(n_embed, n_voacb)
        self.rope = RoPE(60, 32)
        self.layers = nn.Sequential(*[Decoder(n_heads, kv_heads, n_embed, head_dim, hidden_size, max_len, self.rope, eps, device) for _ in range(n_layers)])

    def forward(self, x):
        x = self.embedding(x)
        x = self.layers(x)
        logits = self.lm_head(x)
        return logits

sample = torch.randint(high=32, size=(4, 16)).to(device)
model = SmolLM(32, 180, 30, 9, 3, 20, 540, 32).to(device)
out = model(sample)
print(out.shape)