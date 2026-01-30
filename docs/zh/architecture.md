# 模型架构详解

本文档深入解析 LLM Foundry 的模型架构,帮助您理解现代 Transformer 语言模型的工作原理。

## 📐 整体架构

LLM Foundry 实现了一个 **Decoder-Only Transformer** 架构,类似于 GPT 和 LLaMA 系列模型。

### 架构图

```
输入文本 "Hello World"
    ↓
[Tokenizer] → [101, 2345, 3456]
    ↓
[Token Embedding] → (batch, seq_len, dim)
    ↓
[Transformer Block 1]
    ├─ RMSNorm
    ├─ Causal Self-Attention (with RoPE)
    └─ RMSNorm + MLP (SwiGLU)
    ↓
[Transformer Block 2]
    ...
    ↓
[Transformer Block N]
    ↓
[RMSNorm]
    ↓
[Output Projection] → (batch, seq_len, vocab_size)
    ↓
[Softmax] → 概率分布
    ↓
生成下一个 token
```

---

## 🧱 核心组件

### 1. Token Embedding

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

---

### 2. RMSNorm (Root Mean Square Normalization)

均方根归一化,比 LayerNorm 更高效。

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

**为什么使用 RMSNorm?**

| 特性 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 计算复杂度 | 高 (mean + std) | 低 (只需 RMS) |
| 参数 | 2×dim | 1×dim |
| 性能 | 基线 | 提升 5-10% |
| 稳定性 | 好 | 好 |

**数学公式:**

```
LayerNorm: y = (x - mean(x)) / std(x) * γ + β
RMSNorm:   y = x / RMS(x) * γ
其中 RMS(x) = sqrt(mean(x²) + ε)
```

---

### 3. RoPE (Rotary Position Embedding)

旋转位置编码,通过旋转变换注入位置信息。

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

**为什么使用 RoPE?**

| 方法 | 优点 | 缺点 |
|------|------|------|
| 绝对位置编码 | 简单 | 长度外推能力差 |
| 相对位置编码 | 灵活 | 计算复杂 |
| **RoPE** | **外推能力强** | **实现简单** |

**核心思想:**

通过旋转矩阵对 query 和 key 进行变换,使得注意力分数隐式地包含相对位置信息。

```python
# 预计算旋转频率
θ = 10000^(-2i/d) for i in [0, d/2)
freqs = [θ₀, θ₁, ..., θ_{d/2-1}]

# 对于位置 m
m × freqs = [m×θ₀, m×θ₁, ..., m×θ_{d/2-1}]

# 转换为复数
e^(i×m×θⱼ) = cos(m×θⱼ) + i×sin(m×θⱼ)
```

**优势:**
- ✅ 相对位置编码
- ✅ 长序列外推能力强
- ✅ 不增加参数
- ✅ 计算高效

---

### 4. Grouped Query Attention (GQA)

分组查询注意力,在 MHA 和 MQA 之间取得平衡。

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

**注意力机制对比:**

| 类型 | Query Heads | KV Heads | 参数 | 速度 | 质量 |
|------|-------------|----------|------|------|------|
| MHA | 8 | 8 | 最多 | 慢 | 最好 |
| **GQA** | **8** | **4** | **中等** | **中等** | **好** |
| MQA | 8 | 1 | 最少 | 快 | 可接受 |

**示例配置:**

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

**优势:**
- ✅ 减少 KV Cache 大小 (推理时重要)
- ✅ 降低参数量和计算量
- ✅ 质量损失很小
- ✅ 适合大规模模型

---

### 5. Scaled Dot-Product Attention

缩放点积注意力,使用 PyTorch 优化实现。

```python
# 使用 PyTorch 2.0+ 的高效实现
output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=None,
    dropout_p=dropout if training else 0.0,
    is_causal=True  # 因果掩码
)
```

**手动实现 (教学用):**

```python
# 1. 计算注意力分数
scores = query @ key.transpose(-2, -1)  # (B, H, S, S)
scores = scores / sqrt(head_dim)         # 缩放

# 2. 因果掩码 (下三角)
mask = torch.triu(torch.ones(S, S) * float('-inf'), diagonal=1)
scores = scores + mask  # 屏蔽未来位置

# 3. Softmax
probs = F.softmax(scores, dim=-1)
probs = dropout(probs)

# 4. 加权求和
output = probs @ value  # (B, H, S, D_h)
```

**因果掩码示例:**

```
输入序列: ["The", "cat", "sat"]

注意力矩阵 (允许看到的位置):
       The  cat  sat
The  [  1    0    0  ]
cat  [  1    1    0  ]
sat  [  1    1    1  ]

→ "cat" 只能看到 "The" 和 "cat"
→ "sat" 可以看到所有前面的词
```

---

### 6. MLP with SwiGLU

前馈网络,使用 SwiGLU 激活函数。

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

**激活函数对比:**

| 激活函数 | 公式 | 性能 |
|---------|------|------|
| ReLU | max(0, x) | 基线 |
| GELU | x × Φ(x) | +1-2% |
| Swish | x × σ(x) | +1-2% |
| **SwiGLU** | **Swish(xW₁) ⊙ xW₃** | **+2-3%** |

**为什么是 2/3?**

标准 FFN: `4d → 4d`
SwiGLU 需要两个门: `4d → 2×(8d/3) ≈ 5.33d`

为了保持参数量相近:
`4d → 2×(8d/3) → 4d` ≈ `4d → 4d → 4d`

---

### 7. Transformer Block

完整的 Transformer 层,使用 Pre-Normalization。

```python
class Block(nn.Module):
    def forward(self, x, freqs_cis):
        # Pre-normalization
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
```

**Pre-Norm vs Post-Norm:**

```
Post-Norm (原始 Transformer):
x → Attention → Add → Norm → FFN → Add → Norm

Pre-Norm (现代 LLM):
x → Norm → Attention → Add → Norm → FFN → Add
```

**为什么使用 Pre-Norm?**
- ✅ 训练更稳定
- ✅ 不需要学习率预热
- ✅ 可以训练更深的模型
- ✅ 梯度流动更好

---

## 🔢 参数量计算

### Small 配置示例

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

## 🎯 设计决策

### 1. 为什么选择 Decoder-Only?

| 架构 | 用途 | 优点 | 缺点 |
|------|------|------|------|
| Encoder-Only (BERT) | 理解任务 | 双向上下文 | 不能生成 |
| Encoder-Decoder (T5) | 翻译 | 灵活 | 复杂 |
| **Decoder-Only (GPT)** | **生成** | **简单、通用** | **单向** |

Decoder-Only 架构:
- ✅ 统一的训练目标(下一词预测)
- ✅ 可以处理所有 NLP 任务
- ✅ 架构简单,易于扩展
- ✅ 最适合大规模预训练

### 2. 为什么选择这些现代技术?

| 技术 | 替代方案 | 选择原因 |
|------|---------|---------|
| RMSNorm | LayerNorm | 更快,参数更少 |
| RoPE | 绝对位置编码 | 外推能力强 |
| GQA | MHA | KV Cache 更小 |
| SwiGLU | ReLU/GELU | 性能更好 |
| Pre-Norm | Post-Norm | 训练更稳定 |

这些技术在 LLaMA、Mistral 等最新模型中被广泛采用。

---

## 📊 与其他模型对比

### 架构对比

| 特性 | GPT-2 | LLaMA | **LLM Foundry** |
|------|-------|-------|-----------------|
| Normalization | LayerNorm | RMSNorm | ✅ RMSNorm |
| Position Encoding | Learned | RoPE | ✅ RoPE |
| Attention | MHA | GQA | ✅ GQA |
| Activation | GELU | SwiGLU | ✅ SwiGLU |
| Norm Position | Post | Pre | ✅ Pre |

LLM Foundry 采用了最新的最佳实践!

---

## 🔍 代码导航

**模型实现位置:**

- 完整模型: [src/llm_foundry/models/transformer.py](../../src/llm_foundry/models/transformer.py:1)
- 组件: [src/llm_foundry/models/components.py](../../src/llm_foundry/models/components.py:1)
  - RMSNorm: 第 17 行
  - RoPE: 第 41 行
  - Attention: 第 95 行
  - MLP: 第 181 行
  - Block: 第 213 行

---

## 💡 下一步

- 📖 学习如何训练模型 → [训练指南](training.md)
- 🔧 自定义模型配置 → [配置系统](configuration.md)
- 🚀 优化推理性能 → [推理优化](production/optimization.md)

---

## 📚 延伸阅读

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - RoPE
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) - RMSNorm
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) - GQA, SwiGLU
