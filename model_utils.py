import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


activation_functions = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
}


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim
        self.qkv = nn.Linear(dim, dim * 3)
        self.fc_out = nn.Linear(dim, dim)

    def forward(self, x):
        batch_size, seq_length, dim = x.size()
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, dim)

        out = self.fc_out(attn_output)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, act="relu"):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.act = activation_functions.get(act.lower(), nn.ReLU)()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, act="relu"):
        super(TransformerLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(dim, num_heads)
        self.feed_forward = FeedForward(dim, mlp_dim, act)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_output = self.self_attn(self.ln1(x))
        x = x + attn_output
        ff_output = self.feed_forward(self.ln2(x))
        x = x + ff_output
        return x
