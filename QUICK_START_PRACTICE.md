# LLM Foundry 快速实践指南

> 从零到一，30分钟掌握项目核心

---

## 🎯 学习目标

完成本指南后，你将能够：
- ✅ 理解项目的整体架构
- ✅ 训练你的第一个语言模型
- ✅ 生成文本并理解采样策略
- ✅ 修改配置并观察效果
- ✅ 阅读和理解核心代码

---

## 📋 前置准备

### 环境检查

```bash
# 1. 检查Python版本 (需要 >= 3.8)
python --version

# 2. 检查PyTorch和GPU
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# 3. 检查依赖包
pip list | findstr "torch numpy sentencepiece"
```

**你的环境:**
- ✅ Python 3.11.14
- ✅ PyTorch 2.11.0.dev (CUDA 12.8)
- ✅ CUDA 可用
- ✅ 所有依赖已安装

---

## 🚀 实践步骤

### 步骤 1: 理解项目结构 (5分钟)

```
llm-foundry/
├── tutorials/              # 🎓 教学脚本 - 从这里开始！
│   ├── config.py          # 配置定义
│   ├── model.py           # 模型实现 ⭐核心
│   ├── tokenizer.py       # 分词器
│   ├── dataloader.py      # 数据加载
│   ├── train.py           # 训练脚本
│   └── generate.py        # 生成脚本
│
├── src/llm_foundry/       # 📦 生产包 - 模块化实现
│   ├── models/            # 模型组件
│   ├── training/          # 训练工具
│   ├── inference/         # 推理工具
│   └── ...
│
├── docs/                  # 📚 文档
│   ├── architecture-components.md    # 组件详解
│   └── architecture-training.md      # 训练体系
│
├── examples/              # 💡 使用示例
└── tests/                 # 🧪 测试用例
```

**关键理解:**
- `tutorials/` = 教学版，单文件完整实现，详细注释
- `src/` = 工程版，模块化设计，简洁高效
- 两者功能对等，只是组织方式不同

---

### 步骤 2: 训练第一个模型 (10分钟)

#### 2.1 进入教程目录

```bash
cd tutorials
```

#### 2.2 运行训练脚本

```bash
python train.py
```

**预期输出:**
```
============================================================
初始化训练配置
============================================================
训练配置:
  Batch Size:      32
  Learning Rate:   0.0003
  Max Iterations:  1000
  Eval Interval:   50

使用设备: cuda
  GPU: NVIDIA GeForce RTX 5060
  显存: 8.0 GB

============================================================
初始化数据和模型
============================================================
下载数据...
训练分词器...
✓ 分词器训练完成

模型统计:
  总参数量:       2.08M
  可训练参数:     2.08M
  词汇表大小:     8192
  模型维度:       256
  层数:           4
  注意力头数:     8

============================================================
开始训练
============================================================
step    0 | train loss 9.1234 | val loss 9.2345 | time 0.5s
step   50 | train loss 5.6789 | val loss 5.7890 | time 15.2s
step  100 | train loss 4.2345 | val loss 4.3456 | time 30.1s
...
step  999 | train loss 2.1234 | val loss 2.2345 | time 120.5s

============================================================
训练完成!
============================================================
总耗时: 120.50s (2.0 分钟)
平均速度: 8.3 steps/s

保存模型...
✓ 模型已保存至 minillm.pt
```

**观察要点:**
1. 训练损失应该持续下降
2. 验证损失应该接近训练损失
3. GPU训练约2-5分钟，CPU约10-30分钟

#### 2.3 理解训练过程

打开 `train.py` 阅读代码，重点关注：

```python
# 训练循环的核心
for iter in range(cfg.max_iters):
    # 1. 获取批次数据
    xb, yb = loader.get_batch('train')
    
    # 2. 前向传播
    logits, loss = model(xb, yb)
    
    # 3. 反向传播
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    # 4. 更新参数
    optimizer.step()
```

**关键概念:**
- **前向传播**: 输入 → 模型 → 输出
- **损失计算**: 比较预测和真实值
- **反向传播**: 计算梯度
- **参数更新**: 根据梯度调整权重

---

### 步骤 3: 生成文本 (5分钟)

#### 3.1 运行生成脚本

```bash
python generate.py
```

**预期输出:**
```
============================================================
加载模型和分词器
============================================================
✓ 模型加载成功
✓ 分词器加载成功

============================================================
文本生成示例
============================================================

--- 示例 1: 基础生成 ---
提示词: 满纸荒唐言，
生成: 满纸荒唐言，一把辛酸泪。都云作者痴，谁解其中味...

--- 示例 2: 调整温度 ---
提示词: 人工智能
温度 0.3: 人工智能是计算机科学的一个分支...
温度 0.8: 人工智能正在改变我们的生活方式...
温度 1.5: 人工智能未来将会带来更多惊喜...

--- 示例 3: Top-k 采样 ---
提示词: 深度学习
Top-k=10: 深度学习是机器学习的一个重要分支...
Top-k=50: 深度学习技术在各个领域都有广泛应用...
```

#### 3.2 理解生成过程

打开 `generate.py` 阅读代码，重点关注：

```python
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """自回归生成"""
    for _ in range(max_new_tokens):
        # 1. 获取预测
        logits, _ = model(idx)
        logits = logits[:, -1, :] / temperature
        
        # 2. Top-k 采样
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        # 3. 采样下一个token
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # 4. 拼接到序列
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx
```

**关键概念:**
- **自回归**: 一个token一个token地生成
- **Temperature**: 控制随机性，低温度更确定
- **Top-k**: 只从概率最高的k个token中采样
- **Top-p**: 核采样，累积概率达到p

---

### 步骤 4: 理解模型架构 (10分钟)

#### 4.1 阅读模型代码

打开 `model.py`，这是最核心的文件！

**关键组件:**

1. **RMSNorm** (第66-95行)
```python
class RMSNorm(nn.Module):
    """均方根归一化，比LayerNorm更高效"""
    def forward(self, x):
        # 计算RMS并归一化
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        return self.weight * x_norm
```

2. **RoPE** (第119-161行)
```python
def apply_rotary_emb(xq, xk, freqs_cis):
    """旋转位置编码，通过旋转注入位置信息"""
    # 将实数转为复数
    xq_ = torch.view_as_complex(xq.reshape(..., -1, 2))
    xk_ = torch.view_as_complex(xk.reshape(..., -1, 2))
    
    # 旋转
    xq_out = torch.view_as_real(xq_ * freqs_cis)
    xk_out = torch.view_as_real(xk_ * freqs_cis)
    
    return xq_out, xk_out
```

3. **Attention** (第200-350行)
```python
class CausalSelfAttention(nn.Module):
    """因果自注意力 + GQA"""
    def forward(self, x, freqs_cis):
        # Q, K, V 投影
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        # 应用RoPE
        q, k = apply_rotary_emb(q, k, freqs_cis)
        
        # 注意力计算
        output = F.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        
        return self.wo(output)
```

4. **MLP** (第380-420行)
```python
class MLP(nn.Module):
    """前馈网络 + SwiGLU"""
    def forward(self, x):
        # SwiGLU: Swish(xW1) ⊙ xW3 → W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

5. **Transformer Block** (第450-490行)
```python
class Block(nn.Module):
    """完整的Transformer层"""
    def forward(self, x, freqs_cis):
        # Pre-normalization
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
```

#### 4.2 可视化理解

```
输入 tokens (batch, seq_len)
    ↓
Token Embedding (batch, seq_len, dim)
    ↓
┌─────────────────────────────────┐
│  Transformer Block × N          │
│  ┌───────────────────────────┐  │
│  │ RMSNorm                   │  │
│  │ ↓                         │  │
│  │ Attention (with RoPE)     │  │
│  │ ↓                         │  │
│  │ Residual Add              │  │
│  │ ↓                         │  │
│  │ RMSNorm                   │  │
│  │ ↓                         │  │
│  │ MLP (SwiGLU)              │  │
│  │ ↓                         │  │
│  │ Residual Add              │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
    ↓
Output Projection (batch, seq_len, vocab_size)
    ↓
Logits → Softmax → Probabilities
```

---

### 步骤 5: 实验和修改 (自由探索)

#### 实验 1: 调整模型大小

编辑 `config.py`:

```python
# 原始配置 (Small)
@dataclass
class ModelConfig:
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 4

# 尝试更小的配置 (Tiny)
@dataclass
class ModelConfig:
    dim: int = 128
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2

# 或更大的配置 (Medium)
@dataclass
class ModelConfig:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
```

**观察:**
- 参数量如何变化？
- 训练速度如何变化？
- 生成质量如何变化？

#### 实验 2: 调整训练参数

编辑 `config.py`:

```python
@dataclass
class TrainConfig:
    batch_size: int = 32        # 尝试 16, 64
    learning_rate: float = 3e-4  # 尝试 1e-4, 5e-4
    max_iters: int = 1000        # 尝试 500, 2000
```

**观察:**
- 学习率过高/过低会怎样？
- batch_size 对训练的影响？
- 训练更久是否效果更好？

#### 实验 3: 调整生成参数

编辑 `generate.py`:

```python
# 尝试不同的采样策略
generated = generate(
    model, idx,
    max_new_tokens=100,
    temperature=0.8,    # 尝试 0.3, 1.0, 1.5
    top_k=50,           # 尝试 10, 20, None
    top_p=0.9           # 尝试 0.8, 0.95, None
)
```

**观察:**
- Temperature 如何影响多样性？
- Top-k 和 Top-p 的区别？
- 如何平衡质量和多样性？

#### 实验 4: 使用自己的数据

创建 `my_data.txt`:

```text
你想训练的任何文本...
可以是诗歌、小说、代码、对话...
```

修改 `train.py`:

```python
# 修改数据路径
loader = DataLoader(
    file_path='my_data.txt',  # 使用你的数据
    batch_size=cfg.batch_size,
    block_size=model_cfg.max_seq_len,
    device=device
)
```

---

## 📚 深入学习路径

### 阶段 1: 理解基础 (已完成 ✅)
- [x] 运行训练和生成
- [x] 理解训练循环
- [x] 理解生成过程
- [x] 理解模型架构

### 阶段 2: 深入组件 (推荐)
- [ ] 阅读 `docs/architecture-components.md`
- [ ] 理解 RMSNorm 的数学原理
- [ ] 理解 RoPE 的旋转机制
- [ ] 理解 GQA 的参数共享
- [ ] 理解 SwiGLU 的门控机制

### 阶段 3: 掌握训练 (进阶)
- [ ] 阅读 `docs/architecture-training.md`
- [ ] 理解预训练流程
- [ ] 理解监督微调 (SFT)
- [ ] 理解强化学习 (RLHF)
- [ ] 理解评估和部署

### 阶段 4: 实践项目 (高级)
- [ ] 在自己的数据上训练
- [ ] 实现新的采样策略
- [ ] 添加新的模型组件
- [ ] 优化训练速度
- [ ] 实现模型量化

---

## 🔧 常见问题

### Q1: 训练很慢怎么办？

**A:** 
1. 确认使用GPU: `python -c "import torch; print(torch.cuda.is_available())"`
2. 减小模型: `dim=128, n_layers=2`
3. 减小batch_size: `batch_size=16`
4. 减少训练步数: `max_iters=500`

### Q2: 内存不足怎么办？

**A:**
1. 减小batch_size: `batch_size=8`
2. 减小序列长度: `max_seq_len=128`
3. 减小模型: `dim=128, n_layers=2`

### Q3: 生成质量不好怎么办？

**A:**
1. 训练更久: `max_iters=2000`
2. 使用更大模型: `dim=512, n_layers=8`
3. 使用更多数据
4. 调整采样参数: `temperature=0.8, top_k=50`

### Q4: 如何保存和加载模型？

**A:**
```python
# 保存
torch.save(model.state_dict(), 'my_model.pt')

# 加载
model = MiniLLM(cfg)
model.load_state_dict(torch.load('my_model.pt'))
model.eval()
```

### Q5: 如何使用自己的数据？

**A:**
1. 准备文本文件 (UTF-8编码)
2. 修改 `DataLoader` 的 `file_path` 参数
3. 如果数据量小，减小 `vocab_size`
4. 重新训练分词器和模型

---

## 📊 学习检查清单

### 基础理解 ✅
- [ ] 理解 Transformer 的基本架构
- [ ] 能解释训练循环的四个步骤
- [ ] 理解自回归生成的过程
- [ ] 知道如何调整基本参数

### 组件理解 🎯
- [ ] 能解释 RMSNorm 的作用
- [ ] 理解 RoPE 如何编码位置
- [ ] 知道 GQA 如何减少参数
- [ ] 理解 SwiGLU 的门控机制

### 实践能力 💪
- [ ] 能独立训练一个模型
- [ ] 能调整配置并观察效果
- [ ] 能使用不同采样策略
- [ ] 能在自己的数据上训练

### 高级技能 🚀
- [ ] 能阅读和理解核心代码
- [ ] 能修改模型架构
- [ ] 能优化训练流程
- [ ] 能实现新功能

---

## 🎯 下一步建议

### 如果你是初学者
1. ✅ 完成本指南的所有实践
2. 📚 阅读 `USER_GUIDE.md` 系统学习
3. 📖 阅读 `docs/architecture-components.md`
4. 💡 尝试 `examples/` 中的示例
5. 🤝 参与社区讨论

### 如果你是开发者
1. ✅ 理解 tutorials/ 的实现
2. 📦 学习 src/ 的模块化设计
3. 🧪 编写测试用例
4. 🔧 贡献代码改进
5. 📝 完善文档

### 如果你是研究者
1. ✅ 深入理解架构细节
2. 🔬 设计实验验证想法
3. 📊 对比不同方法
4. 📄 分享研究成果
5. 🤝 与社区交流

---

## 📞 获取帮助

- 📚 查看文档: [docs/](docs/)
- 🐛 提交Issue: GitHub Issues
- 💬 讨论交流: GitHub Discussions
- 📧 查看 CONTRIBUTING.md

---

**祝你学习愉快！🚀**

*通过实践掌握LLM实现，从基础到精通。*

