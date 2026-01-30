"""Transformer 模型完整实现 - 教学版本

这个文件包含了一个完整的 Transformer 语言模型的所有组件。
作为教学材料,所有代码集中在单个文件中,便于理解整体架构。

教学要点:
1. 从底层组件(RMSNorm, RoPE)到完整模型的构建过程
2. 现代 Transformer 的关键技术: GQA, SwiGLU, Pre-normalization
3. 如何将各个组件组装成可训练的模型

架构概览:
    输入 tokens
        ↓
    Token Embedding
        ↓
    ┌─────────────────┐
    │  Transformer    │  ← 重复 n_layers 次
    │    Block        │
    │  ┌──────────┐   │
    │  │ RMSNorm  │   │
    │  │    ↓     │   │
    │  │Attention │   │  (包含 RoPE 和 GQA)
    │  │    ↓     │   │
    │  │   Add    │   │  (残差连接)
    │  └──────────┘   │
    │  ┌──────────┐   │
    │  │ RMSNorm  │   │
    │  │    ↓     │   │
    │  │   MLP    │   │  (SwiGLU)
    │  │    ↓     │   │
    │  │   Add    │   │  (残差连接)
    │  └──────────┘   │
    └─────────────────┘
        ↓
    Final RMSNorm
        ↓
    Output Projection
        ↓
    Logits

建议学习顺序:
1. RMSNorm: 理解归一化
2. RoPE 函数: 理解位置编码
3. CausalSelfAttention: 理解注意力机制和 GQA
4. MLP: 理解 SwiGLU
5. Block: 理解如何组合子层
6. MiniLLM: 理解完整模型架构

与工程版本的区别:
- 工程版本 (src/llm_foundry/): 模块化,每个组件独立文件
- 教学版本 (本文件): 单文件,便于查看完整流程
- 功能完全相同,只是组织方式不同
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig


# ============================================================================
# 第一部分: 基础组件
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (均方根层归一化)

    教学要点:
    - RMSNorm 是 LayerNorm 的简化版本
    - 只使用 RMS 归一化,不减均值
    - 计算更高效,效果相当

    数学公式:
        RMS = sqrt(mean(x²))
        output = (x / RMS) * weight

    与 LayerNorm 对比:
        LayerNorm: (x - mean) / std
        RMSNorm:   x / RMS  (更简单)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """初始化 RMSNorm 层

        Args:
            dim: 特征维度
            eps: 数值稳定性常数,防止除零
        """
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数,初始化为全1
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """前向传播

        Args:
            x: 输入张量,形状 (batch, seq_len, dim)

        Returns:
            归一化后的张量,形状与输入相同
        """
        # 步骤 1: 计算均方 (mean of squares)
        var = torch.mean(x ** 2, dim=-1, keepdim=True)  # (batch, seq_len, 1)

        # 步骤 2: 归一化 (使用 rsqrt 更高效)
        # rsqrt(x) = 1/sqrt(x)
        x_norm = x * torch.rsqrt(var + self.eps)

        # 步骤 3: 应用可学习的缩放
        return self.weight * x_norm


# ============================================================================
# 第二部分: RoPE 位置编码
# ============================================================================

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """预计算 RoPE 的旋转频率

    教学要点:
    - RoPE 使用旋转变换来编码位置信息
    - 不同维度使用不同的旋转频率
    - 预计算可以提高效率

    RoPE 原理:
    - 将特征向量视为复数
    - 根据位置旋转不同角度
    - 使得 q·k 自然包含相对位置信息

    Args:
        dim: 每个注意力头的维度
        end: 最大序列长度
        theta: 频率基数 (默认 10000)

    Returns:
        freqs_cis: 复数频率张量,形状 (end, dim // 2)
    """
    # 步骤 1: 计算每个维度的基础频率
    # θ_i = 1 / (10000^(2i/d))
    # 对于偶数维度 i=0,2,4,...
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 结果: (dim // 2,) - 频率递减

    # 步骤 2: 生成位置索引
    t = torch.arange(end, device=freqs.device)  # (end,) - [0, 1, 2, ..., end-1]

    # 步骤 3: 计算每个位置的旋转角度
    # angle[pos, dim] = pos * freq[dim]
    freqs = torch.outer(t, freqs).float()  # (end, dim // 2)

    # 步骤 4: 转换为复数形式 e^(i*angle)
    # polar(r, θ) 创建 r * e^(iθ)
    # 这里 r=1,所以就是 e^(iθ) = cos(θ) + i*sin(θ)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis):
    """对 query 和 key 应用旋转位置编码

    教学要点:
    - 将实数向量视为复数,进行旋转
    - 只旋转 Q 和 K,不旋转 V
    - 旋转使得点积包含位置信息

    为什么有效?
    - 位置 m 的 q 旋转 θ_m
    - 位置 n 的 k 旋转 θ_n
    - q_m · k_n 包含 (θ_m - θ_n),即相对位置

    Args:
        xq: Query 张量,形状 (batch, seq_len, n_heads, head_dim)
        xk: Key 张量,形状 (batch, seq_len, n_heads, head_dim)
        freqs_cis: 频率张量,形状 (seq_len, head_dim // 2)

    Returns:
        旋转后的 (xq, xk),形状不变
    """
    # 步骤 1: 将实数张量转换为复数
    # 每两个相邻的实数 (a, b) 变成复数 a + bi
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # 结果形状: (batch, seq_len, n_heads, head_dim // 2) 复数

    # 步骤 2: 广播频率以匹配 batch 和 head 维度
    # freqs_cis: (seq_len, head_dim // 2)
    #         -> (1, seq_len, 1, head_dim // 2)
    freqs_cis = freqs_cis.view(1, xq_.size(1), 1, xq_.size(-1))

    # 步骤 3: 复数乘法 = 旋转
    # 在复平面上,乘以 e^(iθ) 就是旋转 θ 角度
    xq_rotated = xq_ * freqs_cis
    xk_rotated = xk_ * freqs_cis

    # 步骤 4: 转换回实数表示
    xq_out = torch.view_as_real(xq_rotated).flatten(3)
    xk_out = torch.view_as_real(xk_rotated).flatten(3)

    # 恢复原始数据类型
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ============================================================================
# 第三部分: 注意力机制
# ============================================================================

class CausalSelfAttention(nn.Module):
    """因果自注意力层 (带 GQA 和 RoPE)

    教学要点:
    1. Multi-Head Attention: 多个注意力头并行计算
    2. GQA: 多个 Q 头共享 KV 头,减少参数和内存
    3. RoPE: 旋转位置编码,不需要额外的位置 embedding
    4. Causal Masking: 只能看到过去的 token

    注意力公式:
        Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

    GQA 示例:
        n_heads=8, n_kv_heads=4
        → 每 2 个 Q 头共享 1 个 KV 头
        → KV cache 减少一半
    """

    def __init__(self, cfg: ModelConfig):
        """初始化注意力层

        Args:
            cfg: 模型配置
        """
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.dim = cfg.dim
        self.head_dim = cfg.dim // cfg.n_heads

        # Q, K, V 投影矩阵
        # 注意: K 和 V 使用 n_kv_heads,实现 GQA
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)

        # 输出投影
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)

        self.dropout = nn.Dropout(cfg.dropout)

        # KV Cache 占位符 (用于生成时的增量解码)
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, freqs_cis):
        """前向传播

        Args:
            x: 输入张量,形状 (batch, seq_len, dim)
            freqs_cis: RoPE 频率,形状 (seq_len, head_dim // 2)

        Returns:
            注意力输出,形状 (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape

        # 步骤 1: 线性投影得到 Q, K, V
        # (batch, seq_len, dim) -> (batch, seq_len, n_heads, head_dim)
        xq = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # 步骤 2: 应用 RoPE 位置编码
        # 只旋转 Q 和 K,不旋转 V
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis[:seq_len])

        # 步骤 3: GQA - 扩展 KV 头以匹配 Q 头数量
        # 如果 n_kv_heads < n_heads,需要重复 KV
        if self.n_kv_heads != self.n_heads:
            # repeat_interleave 在 head 维度上重复
            # 每个 KV 头重复 (n_heads // n_kv_heads) 次
            xk = torch.repeat_interleave(xk, self.n_heads // self.n_kv_heads, dim=2)
            xv = torch.repeat_interleave(xv, self.n_heads // self.n_kv_heads, dim=2)

        # 步骤 4: 转置为注意力计算格式
        # (batch, seq_len, n_heads, head_dim) -> (batch, n_heads, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 步骤 5: 计算注意力
        # 这里展示手动实现的代码(注释掉),帮助理解
        # ===== 手动实现 (教学用) =====
        # scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # # 因果掩码: 上三角为 -inf
        # mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        # scores = scores + mask
        # probs = F.softmax(scores, dim=-1)
        # probs = self.dropout(probs)
        # output = torch.matmul(probs, xv)

        # ===== 高效实现 (实际使用) =====
        # PyTorch 2.0+ 的优化实现,包含:
        # - Scaled dot-product attention
        # - 自动 causal masking
        # - Flash Attention (内存高效)
        output = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=None,  # 不需要显式掩码
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True  # 自动应用因果掩码
        )

        # 步骤 6: 转置回来并合并所有头
        # (batch, n_heads, seq_len, head_dim) -> (batch, seq_len, n_heads * head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # 步骤 7: 输出投影
        return self.wo(output)


# ============================================================================
# 第四部分: 前馈网络
# ============================================================================

class MLP(nn.Module):
    """前馈网络 (使用 SwiGLU 激活函数)

    教学要点:
    - SwiGLU 是 GLU (Gated Linear Unit) 的变体
    - 使用 Swish 激活函数作为门控
    - 比标准的 ReLU FFN 效果更好

    SwiGLU 公式:
        SwiGLU(x) = (Swish(xW1) ⊙ xW3) W2
        其中 Swish(x) = x · sigmoid(x)
             ⊙ 表示逐元素乘法

    结构:
        x → W1 → Swish → ⊙ → W2 → output
        └──→ W3 ────────┘

    参数量:
        标准 FFN: 2 * dim * (4*dim) = 8*dim²
        SwiGLU:   3 * dim * (8/3*dim) ≈ 8*dim² (相当)
    """

    def __init__(self, cfg: ModelConfig):
        """初始化 MLP 层

        Args:
            cfg: 模型配置
        """
        super().__init__()
        # 隐藏层维度: 通常是输入维度的 4 倍
        hidden_dim = 4 * cfg.dim

        # SwiGLU 惯例: 调整为 2/3 以保持参数量相当
        hidden_dim = int(2 * hidden_dim / 3)

        # 三个线性层
        self.w1 = nn.Linear(cfg.dim, hidden_dim, bias=False)  # 门控路径
        self.w2 = nn.Linear(hidden_dim, cfg.dim, bias=False)  # 输出投影
        self.w3 = nn.Linear(cfg.dim, hidden_dim, bias=False)  # 值路径

    def forward(self, x):
        """前向传播

        Args:
            x: 输入张量,形状 (batch, seq_len, dim)

        Returns:
            输出张量,形状 (batch, seq_len, dim)
        """
        # SwiGLU 计算:
        # 1. F.silu(self.w1(x)): 门控信号 (Swish 激活)
        # 2. self.w3(x): 值信号
        # 3. 门控 * 值: 逐元素乘法
        # 4. self.w2(...): 投影回原维度
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ============================================================================
# 第五部分: Transformer 块
# ============================================================================

class Block(nn.Module):
    """Transformer 块 (一层 Transformer)

    教学要点:
    - 每个块包含注意力层和 FFN 层
    - 使用 Pre-normalization (归一化在前)
    - 使用残差连接防止梯度消失

    Pre-norm 结构:
        x → Norm → Attention → (+) → Norm → FFN → (+) → output
        └──────────────────────┘     └─────────────┘
               残差连接                  残差连接

    与 Post-norm 对比:
        Post-norm: x → Attention → (+) → Norm
        Pre-norm:  x → Norm → Attention → (+)

    Pre-norm 优势:
        - 训练更稳定
        - 梯度流更好
        - 大模型的标准选择
    """

    def __init__(self, cfg: ModelConfig):
        """初始化 Transformer 块

        Args:
            cfg: 模型配置
        """
        super().__init__()
        # 两个主要子层
        self.attention = CausalSelfAttention(cfg)
        self.feed_forward = MLP(cfg)

        # 每个子层前的归一化
        self.attention_norm = RMSNorm(cfg.dim)
        self.ffn_norm = RMSNorm(cfg.dim)

    def forward(self, x, freqs_cis):
        """前向传播

        Args:
            x: 输入张量,形状 (batch, seq_len, dim)
            freqs_cis: RoPE 频率

        Returns:
            输出张量,形状 (batch, seq_len, dim)
        """
        # 步骤 1: 注意力子层 (Pre-norm + 残差)
        # Norm → Attention → Add
        h = x + self.attention(self.attention_norm(x), freqs_cis)

        # 步骤 2: FFN 子层 (Pre-norm + 残差)
        # Norm → FFN → Add
        out = h + self.feed_forward(self.ffn_norm(h))

        return out


# ============================================================================
# 第六部分: 完整模型
# ============================================================================

class MiniLLM(nn.Module):
    """完整的 Transformer 语言模型

    教学要点:
    - 这是所有组件的完整组装
    - Token Embedding → Transformer Blocks → Output Projection
    - 使用 RoPE 而不是位置 embedding

    模型结构:
        1. Token Embedding: 将 token ID 映射为向量
        2. N 个 Transformer Block: 核心计算
        3. Final Norm: 最后的归一化
        4. Output Projection: 预测下一个 token

    训练目标:
        给定 tokens[:-1],预测 tokens[1:]
        这是因果语言建模 (Causal Language Modeling)
    """

    def __init__(self, cfg: ModelConfig):
        """初始化完整模型

        Args:
            cfg: 模型配置
        """
        super().__init__()
        self.cfg = cfg

        # 1. Token Embedding
        # 将 token ID (整数) 映射为 dim 维向量
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.dim)

        # 2. Transformer Blocks
        # 堆叠 n_layers 个 Transformer 块
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

        # 3. Final Normalization
        self.norm = RMSNorm(cfg.dim)

        # 4. Output Projection
        # 将 dim 维向量映射回词表大小,得到 logits
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        # 可选: 权重绑定 (Weight Tying)
        # 让 embedding 和 output 共享权重,减少参数量
        # GPT-2 和很多现代模型都使用这个技巧
        # self.token_embedding.weight = self.output.weight

        # 预计算 RoPE 频率
        # *2 是留一些缓冲空间
        self.freqs_cis = precompute_freqs_cis(
            self.cfg.dim // self.cfg.n_heads,  # head_dim
            self.cfg.max_seq_len * 2
        )

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化模型权重

        教学要点:
        - 使用正态分布初始化
        - 标准差 0.02 是常用值
        - 好的初始化对训练很重要

        Args:
            module: 要初始化的模块
        """
        if isinstance(module, nn.Linear):
            # 线性层: 正态分布初始化权重
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Embedding 层: 正态分布初始化
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, targets=None):
        """前向传播

        教学要点:
        - 输入是 token IDs
        - 输出是每个位置的 logits (未归一化的概率)
        - 如果提供 targets,计算交叉熵损失

        Args:
            tokens: 输入 token IDs,形状 (batch, seq_len)
            targets: 目标 token IDs,形状 (batch, seq_len),可选

        Returns:
            logits: 预测的 logits,形状 (batch, seq_len, vocab_size)
            loss: 交叉熵损失,标量 (如果提供 targets)
        """
        batch_size, seq_len = tokens.shape

        # 步骤 1: Token Embedding
        # (batch, seq_len) -> (batch, seq_len, dim)
        h = self.token_embedding(tokens)

        # 步骤 2: 获取 RoPE 频率
        # 只需要当前序列长度的频率
        freqs_cis = self.freqs_cis[:seq_len].to(h.device)

        # 步骤 3: 通过所有 Transformer 块
        for layer in self.layers:
            h = layer(h, freqs_cis)

        # 步骤 4: Final Normalization
        h = self.norm(h)

        # 步骤 5: Output Projection
        # (batch, seq_len, dim) -> (batch, seq_len, vocab_size)
        logits = self.output(h)

        # 步骤 6: 计算损失 (如果提供 targets)
        loss = None
        if targets is not None:
            # 展平以符合 CrossEntropyLoss 的输入格式
            # logits: (batch, seq_len, vocab_size) -> (batch*seq_len, vocab_size)
            # targets: (batch, seq_len) -> (batch*seq_len,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (batch*seq_len, vocab_size)
                targets.view(-1)  # (batch*seq_len,)
            )

        return logits, loss

    def get_num_params(self):
        """获取模型参数量

        Returns:
            参数总数
        """
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == '__main__':
    """测试模型的基本功能

    这个测试展示了:
    1. 如何创建模型
    2. 如何计算参数量
    3. 如何进行前向传播
    4. 输入输出的形状
    """
    print("=" * 60)
    print("测试 MiniLLM 模型")
    print("=" * 60)

    # 创建配置和模型
    cfg = ModelConfig()
    model = MiniLLM(cfg)

    # 打印模型信息
    num_params = model.get_num_params()
    print(f"\n模型参数量: {num_params/1e6:.2f}M")
    print(f"模型配置: {cfg.n_layers} 层, {cfg.dim} 维度, {cfg.n_heads} 个头")

    # 创建虚拟输入
    batch_size = 2
    seq_len = 16
    x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    print(f"\n输入形状: {x.shape}")

    # 前向传播
    logits, loss = model(x, x)  # 使用相同的 x 作为 target (仅用于测试)

    print(f"输出 logits 形状: {logits.shape}")
    print(f"损失值: {loss.item():.4f}")

    print("\n" + "=" * 60)
    print("测试通过!")
    print("=" * 60)
