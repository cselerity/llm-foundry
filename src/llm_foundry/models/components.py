"""Transformer 模型组件

包含模型的基本构建块:
- RMSNorm: 均方根层归一化
- RoPE: 旋转位置编码
- CausalSelfAttention: 因果自注意力
- MLP: 前馈网络(使用 SwiGLU)
- Block: Transformer 块
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import ModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (均方根层归一化)

    通过使用均方根对输入进行归一化来稳定训练。
    与 LayerNorm 相比,RMSNorm 不减去均值,计算更高效。

    Args:
        dim: 输入维度
        eps: 数值稳定性的小常数

    Shape:
        - Input: (batch, seq_len, dim)
        - Output: (batch, seq_len, dim)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (batch, seq_len, dim)
        # 计算 RMS: sqrt(mean(x^2))
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        return self.weight * x_norm


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """预计算 RoPE 中使用的复数指数 (cis) 频率张量

    RoPE (Rotary Position Embedding) 通过旋转变换来编码位置信息。
    这个函数预计算所有位置的旋转频率。

    Args:
        dim: 每个注意力头的维度
        end: 最大序列长度
        theta: 频率的基数(默认 10000.0)

    Returns:
        freqs_cis: 复数频率张量,形状 (end, dim // 2)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # (end, dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis):
    """对 query 和 key 应用旋转位置编码 (RoPE)

    Args:
        xq: Query 张量,形状 (batch, seq_len, n_heads, head_dim)
        xk: Key 张量,形状 (batch, seq_len, n_heads, head_dim)
        freqs_cis: 预计算的频率,形状 (seq_len, head_dim // 2)

    Returns:
        xq_out: 应用 RoPE 后的 query
        xk_out: 应用 RoPE 后的 key
    """
    # 重塑为复数: (batch, seq_len, n_heads, head_dim // 2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 广播 freqs_cis 以匹配 batch 和 heads
    # freqs_cis: (seq_len, head_dim // 2) -> (1, seq_len, 1, head_dim // 2)
    freqs_cis = freqs_cis.view(1, xq_.size(1), 1, xq_.size(-1))

    # 旋转
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class CausalSelfAttention(nn.Module):
    """因果自注意力层

    实现了带有以下特性的多头自注意力:
    - Grouped Query Attention (GQA): 使用更少的 KV heads 来减少参数和内存
    - RoPE: 旋转位置编码
    - Causal Masking: 因果掩码,确保只能看到过去的 token

    Args:
        cfg: 模型配置

    Shape:
        - Input: (batch, seq_len, dim)
        - Output: (batch, seq_len, dim)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.dim = cfg.dim
        self.head_dim = cfg.dim // cfg.n_heads

        # 投影层
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)

        self.dropout = nn.Dropout(cfg.dropout)

        # KV Cache 支持 (暂留占位符,用于未来的增量解码)
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, freqs_cis):
        batch_size, seq_len, _ = x.shape

        # (B, S, D) -> (B, S, H, D_h)
        xq = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # 应用 RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis[:seq_len])

        # 如果 n_kv_heads < n_heads (GQA),则重复 KV heads
        if self.n_kv_heads != self.n_heads:
            xk = torch.repeat_interleave(xk, self.n_heads // self.n_kv_heads, dim=2)
            xv = torch.repeat_interleave(xv, self.n_heads // self.n_kv_heads, dim=2)

        # 转置以进行注意力计算: (B, H, S, D_h)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 使用 PyTorch 2.0+ 的高效实现
        # 自动处理因果掩码和 dropout
        output = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True
        )

        # (B, H, S, D_h) -> (B, S, H, D_h) -> (B, S, D)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)


class MLP(nn.Module):
    """前馈网络 (FeedForward Network)

    使用 SwiGLU 激活函数的 FFN。
    SwiGLU 是 Swish 激活函数的门控线性单元变体,
    被证明在 Transformer 中效果更好。

    Args:
        cfg: 模型配置

    Shape:
        - Input: (batch, seq_len, dim)
        - Output: (batch, seq_len, dim)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden_dim = 4 * cfg.dim
        hidden_dim = int(2 * hidden_dim / 3)  # SwiGLU 惯例

        self.w1 = nn.Linear(cfg.dim, hidden_dim, bias=False)  # Gate (门控)
        self.w2 = nn.Linear(hidden_dim, cfg.dim, bias=False)  # Down (降维)
        self.w3 = nn.Linear(cfg.dim, hidden_dim, bias=False)  # Up (升维)

    def forward(self, x):
        # SwiGLU: (Swish(xW1) * xW3) W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    """Transformer 块

    标准的 Transformer 层,包含:
    - 自注意力层(带残差连接和层归一化)
    - 前馈网络层(带残差连接和层归一化)

    使用 Pre-normalization 架构(LLaMA 风格)。

    Args:
        cfg: 模型配置

    Shape:
        - Input: (batch, seq_len, dim)
        - Output: (batch, seq_len, dim)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attention = CausalSelfAttention(cfg)
        self.feed_forward = MLP(cfg)
        self.attention_norm = RMSNorm(cfg.dim)
        self.ffn_norm = RMSNorm(cfg.dim)

    def forward(self, x, freqs_cis):
        # Pre-normalization (前置归一化)
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
