# 核心组件深入解析

> **现代 Transformer 架构的核心构建块**

本文档深入解析 LLM Foundry 中使用的核心组件，包括 RMSNorm、RoPE、GQA、SwiGLU 等现代 Transformer 技术。

---

## 📖 目录

1. [Token Embedding](#1-token-embedding)
2. [RMSNorm](#2-rmsnorm-root-mean-square-normalization)
3. [RoPE](#3-rope-rotary-position-embedding)
4. [Grouped Query Attention (GQA)](#4-grouped-query-attention-gqa)
5. [Scaled Dot-Product Attention](#5-scaled-dot-product-attention)
6. [MLP with SwiGLU](#6-mlp-with-swiglu)
7. [Transformer Block](#7-transformer-block)
8. [代码导航](#8-代码导航)

---

## 1. Token Embedding

将 token ID 映射到高维向量空间。

```python
self.token_embedding = nn.Embedding(vocab_size, dim)
# 输入: (batch, seq_len) - token IDs
# 输出: (batch, seq_len, dim) - embeddings
```

**特点:**
- 可训练的嵌入矩阵
- 维度: `vocab_size × dim`
- 初始化: 正态分布 (mean=0, std=0.02)

**代码位置**: [src/llm_foundry/models/transformer.py](../../../src/llm_foundry/models/transformer.py)

---

## 2. RMSNorm (Root Mean Square Normalization)

均方根归一化，比 LayerNorm 更高效。

### 实现

```python
class RMSNorm(nn.Module):
    def forward(self, x):
        # 计算 RMS
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        # 归一化
        x_norm = x * torch.rsqrt(var + eps)
        # 缩放
        return self.weight * x_norm
```

### 为什么使用 RMSNorm?

| 特性 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 计算复杂度 | 高 (mean + std) | 低 (只需 RMS) |
| 参数 | 2×dim | 1×dim |
| 性能 | 基线 | 提升 5-10% |
| 稳定性 | 好 | 好 |

### 数学公式

```
LayerNorm: y = (x - mean(x)) / std(x) * γ + β
RMSNorm:   y = x / RMS(x) * γ
其中 RMS(x) = sqrt(mean(x²) + ε)
```

### 优势
- ✅ 计算更快（不需要计算均值）
- ✅ 参数更少（只有缩放参数，无偏置）
- ✅ 训练稳定性好
- ✅ 在 LLaMA、Mistral 等模型中验证有效

**代码位置**: [src/llm_foundry/models/components.py:18](../../../src/llm_foundry/models/components.py#L18)

---

## 3. RoPE (Rotary Position Embedding)

旋转位置编码，通过旋转变换注入位置信息。

### 实现

```python
def apply_rotary_emb(xq, xk, freqs_cis):
    # 将实数张量视为复数
    xq_ = torch.view_as_complex(xq.reshape(..., -1, 2))
    xk_ = torch.view_as_complex(xk.reshape(..., -1, 2))

    # 旋转
    xq_out = torch.view_as_real(xq_ * freqs_cis)
    xk_out = torch.view_as_real(xk_ * freqs_cis)

    return xq_out, xk_out
```

### 为什么使用 RoPE?

| 方法 | 优点 | 缺点 |
|------|------|------|
| 绝对位置编码 | 简单 | 长度外推能力差 |
| 相对位置编码 | 灵活 | 计算复杂 |
| **RoPE** | **外推能力强** | **实现简单** |

### 核心思想

通过旋转矩阵对 query 和 key 进行变换，使得注意力分数隐式地包含相对位置信息。

```python
# 预计算旋转频率
θ = 10000^(-2i/d) for i in [0, d/2)
freqs = [θ₀, θ₁, ..., θ_{d/2-1}]

# 对于位置 m
m × freqs = [m×θ₀, m×θ₁, ..., m×θ_{d/2-1}]

# 转换为复数
e^(i×m×θⱼ) = cos(m×θⱼ) + i×sin(m×θⱼ)
```

### 优势
- ✅ 相对位置编码
- ✅ 长序列外推能力强
- ✅ 不增加参数
- ✅ 计算高效

### 工作原理示例

```
位置 0 的 token: 旋转 0°
位置 1 的 token: 旋转 θ
位置 2 的 token: 旋转 2θ
...

两个 token 之间的相对位置通过旋转角度差体现:
pos_i 和 pos_j 的相对位置 = (j-i)×θ
```

**代码位置**: [src/llm_foundry/models/components.py:66](../../../src/llm_foundry/models/components.py#L66)

---

## 4. Grouped Query Attention (GQA)

分组查询注意力，在 MHA 和 MQA 之间取得平衡。

### 实现

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        self.n_heads = 8      # Query heads
        self.n_kv_heads = 4   # Key/Value heads (GQA)

        self.wq = nn.Linear(dim, n_heads * head_dim)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim)

    def forward(self, x, freqs_cis):
        # 投影
        q = self.wq(x)  # (B, S, n_heads * head_dim)
        k = self.wk(x)  # (B, S, n_kv_heads * head_dim)
        v = self.wv(x)  # (B, S, n_kv_heads * head_dim)

        # 重复 KV heads 以匹配 Q heads
        k = repeat_kv(k, n_heads // n_kv_heads)
        v = repeat_kv(v, n_heads // n_kv_heads)

        # 注意力计算...
```

### 注意力机制对比

| 类型 | Query Heads | KV Heads | 参数 | 速度 | 质量 |
|------|-------------|----------|------|------|------|
| MHA | 8 | 8 | 最多 | 慢 | 最好 |
| **GQA** | **8** | **4** | **中等** | **中等** | **好** |
| MQA | 8 | 1 | 最少 | 快 | 可接受 |

### 配置示例

```python
# 标准配置 (GQA)
n_heads = 8
n_kv_heads = 4  # 每 2 个 Q 共享 1 个 KV

# MHA (Multi-Head Attention)
n_heads = 8
n_kv_heads = 8  # 每个 Q 有独立的 KV

# MQA (Multi-Query Attention)
n_heads = 8
n_kv_heads = 1  # 所有 Q 共享 1 个 KV
```

### 优势
- ✅ 减少 KV Cache 大小（推理时重要）
- ✅ 降低参数量和计算量
- ✅ 质量损失很小
- ✅ 适合大规模模型

### 为什么 GQA 有效?

Query 需要多样性来捕捉不同的语义特征，但 Key/Value 的共享不会显著影响表达能力。GQA 在两者之间找到了最佳平衡点。

**代码位置**: [src/llm_foundry/models/components.py:140](../../../src/llm_foundry/models/components.py#L140)

---

## 5. Scaled Dot-Product Attention

缩放点积注意力，使用 PyTorch 优化实现。

### PyTorch 实现

```python
# 使用 PyTorch 2.0+ 的高效实现
output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=None,
    dropout_p=dropout if training else 0.0,
    is_causal=True  # 因果掩码
)
```

### 手动实现（教学用）

```python
# 1. 计算注意力分数
scores = query @ key.transpose(-2, -1)  # (B, H, S, S)
scores = scores / sqrt(head_dim)         # 缩放

# 2. 因果掩码（下三角）
mask = torch.triu(torch.ones(S, S) * float('-inf'), diagonal=1)
scores = scores + mask  # 屏蔽未来位置

# 3. Softmax
probs = F.softmax(scores, dim=-1)
probs = dropout(probs)

# 4. 加权求和
output = probs @ value  # (B, H, S, D_h)
```

### 因果掩码示例

```
输入序列: ["The", "cat", "sat"]

注意力矩阵（允许看到的位置）:
       The  cat  sat
The  [  1    0    0  ]
cat  [  1    1    0  ]
sat  [  1    1    1  ]

→ "cat" 只能看到 "The" 和 "cat"
→ "sat" 可以看到所有前面的词
```

### 为什么需要缩放?

```python
# 不缩放的问题:
scores = Q @ K^T  # 值可能很大

# 当 head_dim = 64 时:
# scores 的方差 ≈ 64
# softmax 会导致梯度消失

# 缩放后:
scores = Q @ K^T / sqrt(head_dim)
# scores 的方差 ≈ 1
# softmax 梯度更稳定
```

---

## 6. MLP with SwiGLU

前馈网络，使用 SwiGLU 激活函数。

### 实现

```python
class MLP(nn.Module):
    def __init__(self, cfg):
        hidden_dim = 4 * cfg.dim
        hidden_dim = int(2 * hidden_dim / 3)  # SwiGLU 惯例

        self.w1 = nn.Linear(dim, hidden_dim)  # Gate
        self.w2 = nn.Linear(hidden_dim, dim)  # Down
        self.w3 = nn.Linear(dim, hidden_dim)  # Up

    def forward(self, x):
        # SwiGLU: (Swish(xW1) ⊙ xW3) W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

### 激活函数对比

| 激活函数 | 公式 | 性能 |
|---------|------|------|
| ReLU | max(0, x) | 基线 |
| GELU | x × Φ(x) | +1-2% |
| Swish | x × σ(x) | +1-2% |
| **SwiGLU** | **Swish(xW₁) ⊙ xW₃** | **+2-3%** |

### 为什么是 2/3?

```
标准 FFN: dim → 4×dim → dim
参数量: 8×dim²

SwiGLU 需要两个门: dim → hidden_dim (×2 门) → dim
为了保持参数量相近:
hidden_dim = (8×dim²) / (2×2×dim) = 2×dim

但实际使用 2/3 系数:
hidden_dim = int(2 * (4×dim) / 3) ≈ 2.67×dim
```

### 门控机制的优势

```python
# 标准 FFN:
output = W2(activation(W1(x)))

# SwiGLU (门控):
output = W2(activation(W1(x)) * W3(x))
         ↑                      ↑
         内容变换              门控信号

# 门控允许网络动态选择要传递的信息
```

**代码位置**: [src/llm_foundry/models/components.py:248](../../../src/llm_foundry/models/components.py#L248)

---

## 7. Transformer Block

完整的 Transformer 层，使用 Pre-Normalization。

### 实现

```python
class Block(nn.Module):
    def forward(self, x, freqs_cis):
        # Pre-normalization
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
```

### Pre-Norm vs Post-Norm

```
Post-Norm (原始 Transformer):
x → Attention → Add → Norm → FFN → Add → Norm

Pre-Norm (现代 LLM):
x → Norm → Attention → Add → Norm → FFN → Add
```

### 为什么使用 Pre-Norm?

- ✅ 训练更稳定
- ✅ 不需要学习率预热（或减少预热步数）
- ✅ 可以训练更深的模型
- ✅ 梯度流动更好

### 完整流程图

```
输入 x (batch, seq_len, dim)
    ↓
┌───────────────────────────┐
│  Attention Normalization  │
│  h1 = RMSNorm(x)          │
└───────────────────────────┘
    ↓
┌───────────────────────────┐
│  Causal Self-Attention    │
│  h2 = Attention(h1, RoPE) │
└───────────────────────────┘
    ↓
┌───────────────────────────┐
│  Residual Connection      │
│  h3 = x + h2              │
└───────────────────────────┘
    ↓
┌───────────────────────────┐
│  FFN Normalization        │
│  h4 = RMSNorm(h3)         │
└───────────────────────────┘
    ↓
┌───────────────────────────┐
│  Feed-Forward (SwiGLU)    │
│  h5 = MLP(h4)             │
└───────────────────────────┘
    ↓
┌───────────────────────────┐
│  Residual Connection      │
│  output = h3 + h5         │
└───────────────────────────┘
    ↓
输出 output (batch, seq_len, dim)
```

**代码位置**: [src/llm_foundry/models/components.py:280](../../../src/llm_foundry/models/components.py#L280)

---

## 8. 代码导航

### 主工程实现

- **完整模型**: [src/llm_foundry/models/transformer.py](../../../src/llm_foundry/models/transformer.py)
- **组件实现**: [src/llm_foundry/models/components.py](../../../src/llm_foundry/models/components.py)
  - RMSNorm: 第 18 行
  - RoPE: 第 66 行
  - Attention: 第 140 行
  - MLP: 第 248 行
  - Block: 第 280 行

### 教学实现

- **教学版 Model**: [tutorials/model.py](../../../tutorials/model.py)
  - 详细注释
  - 单文件完整实现
  - 与主工程功能对等

### 测试用例

- **单元测试**: [tests/test_models.py](../../../tests/test_models.py)
  - 组件测试
  - 形状验证
  - 数值正确性

---

## 参数量计算示例

### Small 配置

```python
ModelConfig(
    dim=256,
    n_layers=4,
    n_heads=8,
    n_kv_heads=4,
    vocab_size=8192,
    max_seq_len=256
)
```

**参数分解:**

| 组件 | 参数量 | 公式 |
|------|--------|------|
| Token Embedding | 2.1M | vocab_size × dim |
| **每个 Block** | | |
| - RMSNorm (×2) | 512 | 2 × dim |
| - Attention Q | 65K | dim × (n_heads × head_dim) |
| - Attention K | 33K | dim × (n_kv_heads × head_dim) |
| - Attention V | 33K | dim × (n_kv_heads × head_dim) |
| - Attention Out | 65K | (n_heads × head_dim) × dim |
| - MLP W1 | 87K | dim × hidden_dim |
| - MLP W2 | 87K | hidden_dim × dim |
| - MLP W3 | 87K | dim × hidden_dim |
| **Block 小计** | **458K** | |
| **4 个 Block** | **1.83M** | 458K × 4 |
| Output Layer | 2.1M | dim × vocab_size |
| **总计** | **~2.08M** | |

---

## 相关文档

- [架构概览](README.md) - 整体架构设计
- [训练系统](training-system.md) - LLM 训练完整知识
- [设计决策](design-decisions.md) - 为什么选择这些技术
- [学习路径](../../../LEARNING_PATH.md) - 按步骤学习

---

## 延伸阅读

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - RoPE 论文
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) - RMSNorm 论文
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) - GQA, SwiGLU 应用
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) - GQA 论文

---

**深入理解每个组件，掌握现代 Transformer 的核心技术！** 🚀
