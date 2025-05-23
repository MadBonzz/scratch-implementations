import torch
from torch import nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, n_embed, base, max_len):
        super(PositionalEncoding, self).__init__()
        self.n_embed = n_embed
        self.base = base
        self.max_len = max_len
        self.dims = 1 / torch.pow(base, torch.arange(0, n_embed, 2) / n_embed).unsqueeze(0)
        self.pos = torch.arange(max_len).unsqueeze(1).float()
        self.values = self.pos @ self.dims
        self.sin_vals = torch.sin(self.values)
        self.cos_vals = torch.cos(self.values)
        self.combined = torch.cat([self.sin_vals.unsqueeze(-1), self.cos_vals.unsqueeze(-1)], dim=-1)
        self.register_buffer('pe', self.combined.view(self.max_len, self.n_embed))

    def forward(self, x):
        B, T = x.shape
        return torch.cat([self.pe[:T, :].unsqueeze(0)] * B, dim=0)

class AttentionBlock(nn.Module):
    def __init__(self, embed_size, n_head, dropout):
        super(AttentionBlock, self).__init__()
        self.embed_size = embed_size
        self.n_head = n_head
        self.head_size = self.embed_size // self.n_head
        self.drop = dropout
        self.up_proj = nn.Linear(self.embed_size, self.head_size * self.n_head, bias=False)
        self.attn_vectors = nn.Linear(self.head_size * self.n_head, 3 * self.head_size * self.n_head, bias=False)
        self.down_proj = nn.Linear(self.head_size * self.n_head, self.embed_size, bias=False)
        self.dropout = nn.Dropout(self.drop)
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.xavier_uniform_(self.attn_vectors.weight)
        nn.init.xavier_uniform_(self.down_proj.weight)

    def forward(self, x):
        B, T, C = x.shape
        proj = self.up_proj(x)
        q, k, v = self.attn_vectors(proj).split(self.n_head * self.head_size, dim=2)
        q = q.view(B, T, self.n_head, self.head_size)
        k = k.view(B, T, self.n_head, self.head_size)
        v = v.view(B, T, self.n_head, self.head_size)
        output =  F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop)
        output = output.reshape(B, T, self.n_head * self.head_size)
        return self.dropout(self.down_proj(output))
    
class FFN(nn.Module):
    def __init__(self, n_embed, dropout):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(n_embed, 4 * n_embed, bias=False)
        self.linear2 = nn.Linear(4 * n_embed, n_embed, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
    def forward(self, x):
        return self.dropout(self.linear2(F.gelu(self.dropout(self.linear1(x)))))
    
class LayerNorm(nn.Module):
    def __init__(self, n_embed):
        super(LayerNorm, self).__init__()
        self.n_embed = n_embed
        self.gammas = nn.Parameter(torch.ones(self.n_embed))
        self.betas = nn.Parameter(torch.zeros(self.n_embed))

    def forward(self, x):
        x = F.normalize(x, dim=1)
        x = (x * self.gammas) + self.betas
        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, n_embed, n_head, dropout):
        super(EncoderBlock, self).__init__()
        self.mha = AttentionBlock(n_embed, n_head, dropout)
        self.ffn = FFN(n_embed, dropout)
        self.norm1 = LayerNorm(n_embed)
        self.norm2 = LayerNorm(n_embed)
    def forward(self, x):
        x = self.norm1(x + self.mha(x))
        x = self.norm2(x + self.ffn(x))
        return x


class BERT(nn.Module):
    def __init__(self, n_layers, n_embed, n_head, dropout, max_len, vocab_size, base):
        super(BERT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, n_embed)
        self.pe = PositionalEncoding(n_embed, base, max_len)
        self.layers = nn.Sequential(*[EncoderBlock(n_embed, n_head, dropout) for _ in range(n_layers)])
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.lm_head.weight)

    def forward(self, x):
        embed = self.embedding(x)
        pos_embed = self.pe(x)
        embeddings = embed + pos_embed
        out = self.layers(embeddings)
        return self.lm_head(out)