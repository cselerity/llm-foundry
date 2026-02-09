# LLM Foundry 用户指南

> **从快速上手到深入掌握的完整指南**

本指南将帮助您从零开始使用 LLM Foundry，从快速体验第一个模型到深入理解 Transformer 架构。

---

## 📋 目录

- [快速上手](#快速上手-5-10-分钟)
- [系统学习](#系统学习-10-15-小时)
- [深入理解](#深入理解)
- [实践应用](#实践应用)
- [故障排除](#故障排除)

---

## 快速上手 (5-10 分钟)

### 前置要求

- **Python** 3.8 或更高版本
- **RAM** 至少 4GB
- **GPU** (可选) NVIDIA GPU 用于加速训练

### 安装

```bash
# 克隆仓库
git clone https://github.com/your-org/llm-foundry.git
cd llm-foundry

# 安装依赖
pip install -e .

# 验证安装
python -c "import llm_foundry; print('OK')"
```

### 训练第一个模型

```bash
cd tutorials
python train.py      # 训练模型 (~30 秒)
python generate.py   # 生成文本
```

**预期输出**:
```
使用设备: cuda
模型参数量: 2.08M
step 0: train loss 9.1234, val loss 9.2345
step 50: train loss 5.6789, val loss 5.7890
训练完成，耗时 32.45s
```

### 两种使用模式

**教学模式 (tutorials/)** - 适合学习、教学、快速实验
```bash
cd tutorials
python train.py
python generate.py
```

**包模式 (src/)** - 适合研究、生产、定制化开发
```python
from llm_foundry import ModelConfig, MiniLLM, DataLoader

cfg = ModelConfig(dim=512, n_layers=8)
model = MiniLLM(cfg)
# ...
```

---

## 系统学习 (10-15 小时)

### 第一阶段: 理解核心代码 (2-3 小时)

#### 1. 配置系统
📖 阅读: [tutorials/config.py](tutorials/config.py)
- `ModelConfig`: 模型架构配置
- `TrainConfig`: 训练超参数配置

#### 2. 模型架构 - 核心组件
📖 阅读: [src/llm_foundry/models/components.py](src/llm_foundry/models/components.py)

**RMSNorm** (第18-64行)
- 归一化技术，比 LayerNorm 更高效
- 只需计算 RMS，不需要均值

**RoPE 位置编码** (第66-148行)
- 通过旋转编码位置信息
- 相对位置在点积中自然体现

**注意力机制** (第150-258行)
- Self-Attention 和 GQA (分组查询注意力)
- GQA 减少 KV Cache 大小，降低参数量

**前馈网络** (第260-278行)
- SwiGLU 激活函数
- 门控机制提升性能

**Transformer 块** (第280-312行)
- Pre-normalization 架构
- 残差连接

#### 3. 完整模型
📖 阅读: [src/llm_foundry/models/transformer.py](src/llm_foundry/models/transformer.py)
- 理解如何组装各个组件
- 查看 `forward` 方法的完整流程

#### 4. 数据处理和分词器
📖 阅读:
- [tutorials/data.py](tutorials/data.py) - 数据加载流程
- [tutorials/tokenizer.py](tutorials/tokenizer.py) - BPE 分词原理

---

### 第二阶段: 深入理解 (3-4 小时)

#### 5. 架构深度解析
📖 阅读: [docs/architecture-components.md](docs/architecture-components.md)

**核心组件详解**:
- Token Embedding
- RMSNorm vs LayerNorm
- RoPE 工作原理
- GQA vs MHA vs MQA
- SwiGLU 门控机制
- Pre-Norm vs Post-Norm

#### 6. 训练流程详解
📖 阅读:
- [tutorials/train.py](tutorials/train.py)
- [src/llm_foundry/training/trainer.py](src/llm_foundry/training/trainer.py)
- [docs/architecture-training.md](docs/architecture-training.md)

**训练全流程**:
1. 数据准备 (Data Preparation)
2. 预训练 (Pre-training)
3. 监督微调 (SFT)
4. 奖励建模 (Reward Modeling)
5. 强化学习 (RLHF)
6. 评估与部署 (Evaluation & Deployment)

**关键技术**:
- AdamW 优化器
- Cosine 学习率调度 + Warmup
- 梯度裁剪
- 混合精度训练

#### 7. 推理和生成
📖 阅读:
- [tutorials/generate.py](tutorials/generate.py)
- [src/llm_foundry/inference/generator.py](src/llm_foundry/inference/generator.py)

**生成技术**:
- 自回归生成流程
- Temperature 控制随机性
- Top-k 和 Top-p 采样
- KV Cache 优化

---

### 第三阶段: 实践应用 (4-6 小时)

#### 8. 自定义数据集
📖 阅读: [tutorials/dataloader.py](tutorials/dataloader.py)
🎯 实践: [examples/02_custom_data.py](examples/02_custom_data.py)

**任务**:
- 准备自己的文本数据
- 训练自定义词表
- 在自己的数据上训练模型

#### 9. 调整模型配置
🎯 尝试不同的模型配置

```python
# 小型模型 (快速实验)
small_cfg = ModelConfig(dim=256, n_layers=4, n_heads=8, n_kv_heads=4)

# 中型模型 (更好效果)
medium_cfg = ModelConfig(dim=512, n_layers=8, n_heads=8, n_kv_heads=4)

# RTX 5060 优化配置
rtx5060_cfg = ModelConfig(dim=704, n_layers=10, n_heads=10, n_kv_heads=5)
```

#### 10. 超参数调优
**实验**:
- 调整学习率 (1e-4 到 1e-3)
- 调整 batch size
- 调整 warmup 步数
- 使用 learning rate scheduler

#### 11. 高级生成技巧
🎯 实践: [examples/03_generation_sampling.py](examples/03_generation_sampling.py)

**探索**:
- Temperature 对多样性的影响 (0.1-1.0)
- Top-k 和 Top-p 的平衡
- 组合使用多种策略

---

### 第四阶段: 生产实践 (可选, 6-8 小时)

#### 12. 使用包模式开发
```python
from llm_foundry import (
    ModelConfig, TrainConfig,
    MiniLLM, Tokenizer, DataLoader
)
from llm_foundry.training import Trainer
from llm_foundry.inference import Generator

# 构建完整应用
cfg = ModelConfig()
model = MiniLLM(cfg)
trainer = Trainer(model, train_cfg)
trainer.train()
```

#### 13. 命令行工具
```bash
# 使用配置文件训练
python scripts/train.py --config configs/medium.yaml

# 生成文本
python scripts/generate.py \
    --checkpoint model.pt \
    --prompt "Once upon a time" \
    --temperature 0.8
```

#### 14. 生产部署
📖 阅读: [docs/architecture-training.md](docs/architecture-training.md)

**生产级技术**:
- 分布式训练 (DDP/FSDP)
- 混合精度训练 (FP16/BF16)
- 模型服务 (FastAPI)
- 推理优化 (量化、Flash Attention)

---

## 深入理解

### 核心技术

#### RoPE (Rotary Position Embedding)
通过旋转变换注入位置信息，相对位置在点积中自然体现。

**优势**:
- 长序列外推能力强
- 不增加参数
- 计算高效

#### GQA (Grouped Query Attention)
在 MHA 和 MQA 之间取得平衡，减少 KV Cache 大小。

**配置示例**:
```python
# MHA (Multi-Head Attention)
n_heads = 8, n_kv_heads = 8

# GQA (Grouped Query Attention)
n_heads = 8, n_kv_heads = 4  # 每 2 个 Q 共享 1 个 KV

# MQA (Multi-Query Attention)
n_heads = 8, n_kv_heads = 1  # 所有 Q 共享 1 个 KV
```

#### SwiGLU
高性能激活函数，使用门控机制。

**公式**: `Swish(xW₁) ⊙ xW₃ W₂`

#### RMSNorm
均方根归一化，比 LayerNorm 更高效。

**公式**: `y = x / RMS(x) * γ`

---

## 实践应用

### 自定义配置

#### 方法 1: 编辑配置文件
编辑 `tutorials/config.py`:
```python
@dataclass
class ModelConfig:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    vocab_size: int = 8192
    max_seq_len: int = 512

@dataclass
class TrainConfig:
    batch_size: int = 16
    learning_rate: float = 3e-4
    max_iters: int = 5000
    eval_interval: int = 100
```

#### 方法 2: 使用预设配置
```python
from tutorials.config import (
    get_small_config,      # ~2M 参数
    get_medium_config,     # ~10M 参数
    get_rtx5060_config    # ~70M 参数
)
```

### 硬件选择参考

| 硬件 | 模型规模 | 训练时间* | 指南 |
|------|---------|----------|------|
| CPU | 2M | 10-30min | 使用 small 配置 |
| RTX 5060 (8GB) | 70M | 30-40min | [RTX 5060 指南](docs/hardware-rtx5060.md) |
| Apple M4 Pro | 68M | 40-60min | 自定义配置 |
| RTX 4090 (24GB) | 200M+ | 10-20min | 自定义配置 |

*基于 10k training steps

---

## 故障排除

### 问题 1: 训练太慢

**解决方案**:
1. 检查 GPU: `torch.cuda.is_available()`
2. 减小模型: 降低 `dim` 或 `n_layers`
3. 减小批次: 降低 `batch_size`
4. 使用混合精度: 参考 [训练系统](docs/architecture-training.md)

### 问题 2: CUDA Out of Memory

**解决方案**:
```python
# 减小 batch_size
train_cfg.batch_size = 16

# 减小 max_seq_len
model_cfg.max_seq_len = 128

# 减小模型大小
model_cfg.dim = 256
model_cfg.n_layers = 4
```

### 问题 3: 生成质量不好

**解决方案**:
1. 增加训练步数: `train_cfg.max_iters = 5000`
2. 使用更大的模型: `model_cfg = get_medium_config()`
3. 调整采样参数:
   ```python
   temperature = 0.8   # 降低随机性
   top_k = 50          # 限制候选词
   top_p = 0.9         # 核采样
   ```

### 问题 4: 找不到模块

**解决方案**:
```bash
cd llm-foundry
pip install -e .
python -c "import llm_foundry; print('OK')"
```

---

## 学习检查清单

### 基础概念
- [ ] 理解 Transformer 的基本架构
- [ ] 能解释 Self-Attention 的工作原理
- [ ] 理解因果语言建模的训练目标
- [ ] 知道如何计算模型参数量

### 核心技术
- [ ] 能解释 RoPE 如何编码位置信息
- [ ] 理解 GQA 的参数共享机制
- [ ] 知道 SwiGLU 的门控机制
- [ ] 理解 Pre-normalization 的优势

### 实践能力
- [ ] 能独立训练一个小型模型
- [ ] 能调整模型配置和训练参数
- [ ] 能使用不同采样策略生成文本
- [ ] 能在自己的数据上训练模型

### 高级技能
- [ ] 能阅读和理解核心代码
- [ ] 能编写测试用例
- [ ] 能使用模块化 API 开发应用
- [ ] 理解生产部署的考虑因素

---

## 外部资源

### 经典论文
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
- [RoFormer](https://arxiv.org/abs/2104.09864) - RoPE 论文
- [LLaMA](https://arxiv.org/abs/2302.13971) - 现代 LLM 架构参考
- [GPT-3](https://arxiv.org/abs/2005.14165) - 大规模语言模型

### 学习资源
- [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) - 极简 GPT 实现
- [Stanford CS224N](http://web.stanford.edu/class/cs224n/) - NLP 课程
- [Hugging Face Course](https://huggingface.co/course) - 免费 NLP 课程

---

## 获取帮助

- 📚 **查看文档**: [docs/README.md](docs/README.md)
- 🐛 **提交 Issue**: [GitHub Issues](https://github.com/your-org/llm-foundry/issues)
- 💬 **讨论**: [GitHub Discussions](https://github.com/your-org/llm-foundry/discussions)
- 🤝 **贡献代码**: [贡献指南](CONTRIBUTING.md)

---

**祝您学习愉快！** 🚀
