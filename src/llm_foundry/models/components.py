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
        """前向传播

        RMSNorm 的核心思想:
        1. 计算 RMS (Root Mean Square): sqrt(mean(x²))
        2. 用 RMS 归一化: x / RMS
        3. 应用可学习的缩放参数

        与 LayerNorm 的区别:
        - LayerNorm: (x - mean) / std  (需要计算均值和方差)
        - RMSNorm: x / RMS             (只需计算 RMS,更快)
        """
        # x: (batch, seq_len, dim)

        # Step 1: 计算每个位置的 RMS
        # 对最后一个维度(特征维度)求平方和的均值
        var = torch.mean(x ** 2, dim=-1, keepdim=True)  # (batch, seq_len, 1)

        # Step 2: 归一化
        # rsqrt(x) = 1/sqrt(x),直接计算倒数平方根更高效
        # eps 用于数值稳定,避免除零
        x_norm = x * torch.rsqrt(var + self.eps)  # (batch, seq_len, dim)

        # Step 3: 应用可学习的缩放参数
        # weight 是一个可学习向量,允许网络调整归一化后的尺度
        return self.weight * x_norm


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """预计算 RoPE 中使用的复数指数 (cis) 频率张量

    RoPE (Rotary Position Embedding) 通过旋转变换来编码位置信息。
    这个函数预计算所有位置的旋转频率。

    核心思想:
    对于位置 m 和维度 i,计算旋转角度 θ_i = m / (10000^(2i/d))
    然后转换为复数 e^(iθ) = cos(θ) + i*sin(θ)

    Args:
        dim: 每个注意力头的维度
        end: 最大序列长度
        theta: 频率的基数(默认 10000.0)

    Returns:
        freqs_cis: 复数频率张量,形状 (end, dim // 2)
    """
    # Step 1: 计算每个维度的基础频率
    # 公式: θ_i = 1 / (10000^(2i/d))
    # arange(0, dim, 2) 生成 [0, 2, 4, ...] 用于偶数维度
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 结果形状: (dim // 2,)  每个元素是一个频率

    # Step 2: 为每个位置计算旋转角度
    # 位置 0, 1, 2, ..., end-1
    t = torch.arange(end, device=freqs.device)  # (end,)

    # Step 3: 外积得到所有 (位置, 频率) 的组合
    # 位置 m × 频率 θ_i = 旋转角度
    freqs = torch.outer(t, freqs).float()  # (end, dim // 2)

    # Step 4: 将角度转换为复数 e^(i×angle)
    # torch.polar(r, θ) 创建复数 r × e^(iθ)
    # 这里 r=1,所以就是 e^(iθ) = cos(θ) + i×sin(θ)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis):
    """对 query 和 key 应用旋转位置编码 (RoPE)

    RoPE 的核心思想:
    将特征向量视为复数,通过复数乘法实现旋转。
    这样注意力分数 q·k 就隐式地包含了相对位置信息。

    为什么有效?
    - q_m 在位置 m 旋转 θ_m
    - k_n 在位置 n 旋转 θ_n
    - q_m·k_n 包含 (θ_m - θ_n),即相对位置 (m-n)

    Args:
        xq: Query 张量,形状 (batch, seq_len, n_heads, head_dim)
        xk: Key 张量,形状 (batch, seq_len, n_heads, head_dim)
        freqs_cis: 预计算的频率,形状 (seq_len, head_dim // 2)

    Returns:
        xq_out: 应用 RoPE 后的 query
        xk_out: 应用 RoPE 后的 key
    """
    # Step 1: 将实数张量转换为复数形式
    # 原始: (batch, seq, heads, dim) 最后 dim 是实数
    # 转换: (batch, seq, heads, dim/2) 每个元素是复数
    # 做法: 将相邻两个实数 (a, b) 组合成复数 a + bi
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Step 2: 广播旋转频率以匹配 batch 和 head 维度
    # freqs_cis 形状: (seq_len, head_dim // 2)
    # 需要变成: (1, seq_len, 1, head_dim // 2)
    # 这样可以对每个 batch、每个 head 应用相同的位置编码
    freqs_cis = freqs_cis.view(1, xq_.size(1), 1, xq_.size(-1))

    # Step 3: 复数乘法实现旋转
    # 在复平面上,乘以 e^(iθ) 就是旋转 θ 角度
    # xq_ * freqs_cis 对每个位置的向量应用相应的旋转
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)  # 转回实数
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    # Step 4: 恢复原始数据类型
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

        # 投影层 - GQA (Grouped Query Attention) 架构
        # Q 投影: dim -> n_heads * head_dim (完整的查询头)
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)

        # K, V 投影: dim -> n_kv_heads * head_dim (分组的 KV 头)
        # GQA 关键思想: 多个 query 头共享同一个 key-value 头
        # 例如: 8 个 Q 头可能只使用 4 个 KV 头,每 2 个 Q 头共享 1 个 KV 头
        # 这样可以大幅减少参数量和 KV cache 内存占用
        self.wk = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)

        # 输出投影: 将多头输出合并回原始维度
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)

        self.dropout = nn.Dropout(cfg.dropout)

        # KV Cache 支持 (暂留占位符,用于未来的增量解码)
        # 在生成阶段可以缓存之前的 key 和 value,避免重复计算
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, freqs_cis):
        """前向传播

        注意力机制的完整流程:
        1. Q/K/V 投影
        2. 应用 RoPE 位置编码
        3. GQA: 扩展 KV 头以匹配 Q 头数量
        4. 计算注意力分数并应用因果掩码
        5. 输出投影
        """
        batch_size, seq_len, _ = x.shape

        # Step 1: 线性投影得到 Q, K, V
        # x 形状: (batch, seq_len, dim)
        # 投影后: (batch, seq_len, n_heads * head_dim)
        # 重塑为: (batch, seq_len, n_heads, head_dim)
        xq = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Step 2: 应用旋转位置编码 (RoPE)
        # 这一步为 Q 和 K 添加位置信息,但保持 V 不变
        # RoPE 的优势: 位置信息通过旋转编码在点积中自然表达相对位置
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis[:seq_len])

        # Step 3: GQA - 扩展 KV 头
        # 如果 n_kv_heads < n_heads (例如 4 < 8),则需要重复 KV 头
        # 让多个 Q 头共享同一个 KV 头,实现分组查询
        # 例如: 4 个 KV 头扩展为 8 个,每个 KV 头被使用 2 次
        if self.n_kv_heads != self.n_heads:
            # repeat_interleave 在 dim=2 (head维度) 上重复
            # 每个 KV 头重复 (n_heads // n_kv_heads) 次
            xk = torch.repeat_interleave(xk, self.n_heads // self.n_kv_heads, dim=2)
            xv = torch.repeat_interleave(xv, self.n_heads // self.n_kv_heads, dim=2)

        # Step 4: 转置为注意力计算的标准格式
        # 从 (batch, seq_len, n_heads, head_dim)
        # 到 (batch, n_heads, seq_len, head_dim)
        # 这样可以对每个头并行计算注意力
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Step 5: 计算缩放点积注意力
        # PyTorch 2.0+ 的优化实现,包含:
        # - Scaled dot-product: softmax(Q·K^T / √d_k)·V
        # - Causal masking: 确保位置 i 只能看到 ≤i 的位置
        # - Dropout: 训练时随机丢弃部分注意力连接
        # - Flash Attention: 内存高效的注意力实现
        output = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=None,  # 不需要显式掩码
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True  # 自动应用因果掩码
        )

        # Step 6: 转置回来并合并所有头
        # (batch, n_heads, seq_len, head_dim) -> (batch, seq_len, n_heads, head_dim)
        # -> (batch, seq_len, n_heads * head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Step 7: 输出投影
        # 将多头输出映射回原始维度
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
        # FFN 隐藏层维度通常是输入维度的 4 倍
        hidden_dim = 4 * cfg.dim

        # SwiGLU 惯例: 将隐藏维度调整为原来的 2/3
        # 这是因为 SwiGLU 使用两个并行的线性层 (w1 和 w3)
        # 为了保持总参数量接近标准 FFN,需要减小隐藏维度
        hidden_dim = int(2 * hidden_dim / 3)

        # 三个线性层实现 SwiGLU
        # SwiGLU 公式: SwiGLU(x) = Swish(xW1) ⊙ (xW3)
        # 其中 ⊙ 表示逐元素乘法

        self.w1 = nn.Linear(cfg.dim, hidden_dim, bias=False)  # Gate (门控路径)
        self.w2 = nn.Linear(hidden_dim, cfg.dim, bias=False)  # Down (降维投影)
        self.w3 = nn.Linear(cfg.dim, hidden_dim, bias=False)  # Up (升维路径)

    def forward(self, x):
        """前向传播

        SwiGLU 激活函数的计算过程:
        1. 门控路径: gate = Swish(xW1) = x·σ(x)·W1
        2. 值路径: value = xW3
        3. 门控调制: gate ⊙ value (逐元素乘法)
        4. 降维投影: (gate ⊙ value)W2

        为什么 SwiGLU 有效?
        - 门控机制允许网络动态控制信息流
        - Swish 激活函数 (x·σ(x)) 比 ReLU 更平滑,梯度更好
        - 两路并行设计增强了表达能力
        """
        # Step 1: 计算门控信号
        # F.silu 就是 Swish 激活函数: silu(x) = x * sigmoid(x)
        gate = F.silu(self.w1(x))  # (batch, seq_len, hidden_dim)

        # Step 2: 计算值路径
        value = self.w3(x)  # (batch, seq_len, hidden_dim)

        # Step 3: 门控调制 + 降维投影
        # gate * value: 门控机制,控制有多少信息通过
        # w2(): 投影回原始维度
        return self.w2(gate * value)


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
        # 两个主要子层
        self.attention = CausalSelfAttention(cfg)  # 自注意力层
        self.feed_forward = MLP(cfg)  # 前馈网络

        # Pre-normalization: 在子层之前进行归一化
        # 与 Post-normalization (原始 Transformer) 相比的优势:
        # - 训练更稳定,不容易梯度爆炸
        # - 可以使用更大的学习率
        # - 收敛更快
        self.attention_norm = RMSNorm(cfg.dim)  # 注意力层的归一化
        self.ffn_norm = RMSNorm(cfg.dim)  # FFN 层的归一化

    def forward(self, x, freqs_cis):
        """前向传播

        Pre-normalization + Residual Connection 架构:

        标准流程:
        1. x -> Norm -> Attention -> + x (残差连接)
        2. h -> Norm -> FFN -> + h (残差连接)

        与 Post-normalization (原始 Transformer) 的对比:
        - Post-norm: x -> Attention -> + x -> Norm
        - Pre-norm:  x -> Norm -> Attention -> + x

        为什么 Pre-norm 更好?
        - 梯度流更稳定: 残差路径直接连接,梯度不经过归一化
        - 训练更容易: 不需要 learning rate warmup
        - 大模型首选: GPT-3, LLaMA 等都使用 Pre-norm
        """
        # Step 1: 自注意力子层
        # 归一化 -> 注意力 -> 残差连接
        # self.attention_norm(x): 先归一化输入
        # self.attention(...): 计算注意力
        # + x: 将原始输入加回来(残差连接)
        h = x + self.attention(self.attention_norm(x), freqs_cis)

        # Step 2: 前馈网络子层
        # 归一化 -> FFN -> 残差连接
        # self.ffn_norm(h): 归一化第一个子层的输出
        # self.feed_forward(...): 前馈网络
        # + h: 残差连接
        out = h + self.feed_forward(self.ffn_norm(h))

        return out
