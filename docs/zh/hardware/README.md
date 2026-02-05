# 硬件配置指南

> **针对不同硬件平台的优化配置**

本指南帮助您根据硬件选择最佳的模型配置和训练策略。

---

## 🎯 快速选择

### 我的硬件是...

| 硬件类型 | 指南 | 模型规模 | 训练时间* |
|---------|------|---------|----------|
| **RTX 5060 / 4060 / 3060 (8GB+)** | [RTX 5060 指南](rtx-5060.md) | 70M 参数 | 30-40 分钟 |
| **Apple M4 Pro / M3 Pro** | [Apple Silicon 指南](apple-silicon.md) | 68M 参数 | 40-60 分钟 |
| **RTX 4090 / 3090 (24GB+)** | [高端 GPU 配置](#高端-gpu-配置) | 200M+ 参数 | 10-20 分钟 |
| **CPU 或低端 GPU** | [基础配置](#基础配置) | 2-10M 参数 | 10-30 分钟 |

*基于 10k training steps

---

## 📋 硬件推荐对比

### 完整对比表

| 用途 | GPU/CPU | 显存/内存 | 系统内存 | 模型规模 | 训练时间* | 成本 |
|------|---------|----------|---------|----------|-----------|------|
| **学习** | CPU | - | 8GB | 2M | 10-30min | ¥0 |
| **实验** | RTX 3060 | 12GB | 16GB | 10M | 5-10min | ~¥2000 |
| **本地训练** | **RTX 5060** | **8GB** | **32GB** | **70M** | **30-40min** | **~¥2500** |
| **专业开发** | RTX 4090 | 24GB | 64GB | 200M+ | 10-20min | ~¥15000 |
| **Mac 用户** | **M4 Pro** | **32GB 统一** | - | **68M** | **40-60min** | **~¥18000** |
| **研究** | A100 (40GB) | 40GB | 128GB | 500M+ | 5-10min | 云租用 |

### 性能指标

| 硬件 | Tokens/秒 (训练) | Tokens/秒 (推理) | 功耗 | 性价比 |
|------|-----------------|-----------------|------|--------|
| RTX 5060 | 2500-3500 | 1500-2500 | 120W | ⭐⭐⭐⭐⭐ |
| RTX 4090 | 10000+ | 5000-8000 | 450W | ⭐⭐⭐ |
| M4 Pro | 1500-2500 | 1000-1500 | 50W | ⭐⭐⭐⭐ |
| CPU (i7) | 100-300 | 50-100 | 65W | ⭐⭐ |

---

## 🔧 配置选择

### RTX 5060 (8GB) - 推荐配置

**适合人群:**
- 个人开发者
- 学习者和研究者
- 预算有限但需要 GPU 加速

**查看详细指南:** [RTX 5060 完整指南](rtx-5060.md)

**快速配置:**

```python
from tutorials.config import get_rtx5060_config, get_rtx5060_train_config

model_cfg = get_rtx5060_config()      # 70-75M 参数
train_cfg = get_rtx5060_train_config() # batch_size=24

# 或使用脚本
python tutorials/train_rtx5060.py
```

---

### Apple Silicon (M4 Pro / M3 Pro) - 优化配置

**适合人群:**
- Mac 用户
- 需要低功耗训练
- 移动办公

**查看详细指南:** [Apple Silicon 指南](apple-silicon.md)

**快速配置:**

```python
from tutorials.config import get_m4pro_config, get_m4pro_train_config

model_cfg = get_m4pro_config()        # 68-72M 参数
train_cfg = get_m4pro_train_config()  # batch_size=32, MPS 优化

# 或使用脚本
python tutorials/train_m4pro.py
```

---

### 高端 GPU 配置

**RTX 4090 / 3090 / A100**

```python
from llm_foundry.config import ModelConfig, TrainConfig

# 大型配置 (~200M 参数)
model_cfg = ModelConfig(
    dim=1024,
    n_layers=24,
    n_heads=16,
    n_kv_heads=8,
    vocab_size=32000,
    max_seq_len=2048
)

train_cfg = TrainConfig(
    batch_size=64,        # 大显存可用更大批次
    learning_rate=3e-4,
    max_iters=50000
)
```

---

### 基础配置

**CPU 或低端 GPU (< 4GB)**

```python
from tutorials.config import get_small_config, get_small_train_config

# 小型配置 (~2M 参数)
model_cfg = get_small_config()        # dim=256, layers=4
train_cfg = get_small_train_config()  # batch_size=4

# 训练
cd tutorials
python train.py  # 自动使用 small 配置
```

---

## 📊 内存占用估算

### 模型内存占用公式

```python
# 训练时内存占用 (FP32)
memory_training = params * 4  # 参数 (FP32)
                + params * 4  # 梯度
                + params * 8  # 优化器状态 (AdamW)
                + activations # 激活值 (取决于 batch_size 和 seq_len)

# 示例: 70M 参数模型
params = 70_000_000
base_memory = 70 * (4 + 4 + 8) = 1120 MB (~1.1 GB)
activations ≈ batch_size * seq_len * layers * dim * 4 * 2
            = 24 * 256 * 10 * 704 * 4 * 2
            ≈ 880 MB

total ≈ 2 GB (实际约 3-4 GB，包括框架开销)
```

### 快速估算表

| 模型规模 | 参数量 | FP32 训练 | FP16 训练 | 推理 (FP16) |
|---------|--------|----------|----------|-----------|
| Tiny | 2M | ~500MB | ~300MB | ~100MB |
| Small | 10M | ~2GB | ~1GB | ~500MB |
| Medium | 70M | ~12GB | ~6GB | ~3GB |
| Large | 200M | ~35GB | ~18GB | ~9GB |

---

## ⚡ 优化建议

### 通用优化

1. **使用混合精度 (FP16/BF16)**
   - 减少内存 50%
   - 加速训练 2-3 倍
   - RTX 系列强烈推荐

2. **调整 batch_size**
   - 尽可能大，但不 OOM
   - 使用梯度累积模拟大 batch

3. **优化序列长度**
   - 根据任务需求设置
   - 长序列内存占用更大

### GPU 特定优化

**NVIDIA GPU:**
- ✅ 启用 TF32 (Ampere 及以上)
- ✅ 使用 `torch.compile()` (PyTorch 2.0+)
- ✅ 使用 Flash Attention
- ✅ 使用 cuDNN 自动调优

**Apple Silicon (MPS):**
- ✅ 使用 BF16 而不是 FP16
- ✅ 优化层数为 10 左右 (MPS 友好)
- ✅ 使用统一内存优势 (CPU + GPU 共享)

---

## 🔧 配置速查表

### 快速参考

需要快速查找配置参数？

→ **[配置速查表 (quick-reference.md)](quick-reference.md)**

包含:
- 所有预设配置的参数表
- 命令行快速参考
- 常见问题解决

---

## 🆘 常见问题

### Q: 如何知道我的 GPU 显存？

```bash
# NVIDIA GPU
nvidia-smi

# 或在 Python 中
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

### Q: 我的 GPU 不在列表中怎么办?

**A:** 根据显存选择配置:
- **4GB**: 使用 small 配置
- **8GB**: 使用 RTX 5060 配置
- **12GB**: 可以增大到 100M 参数
- **24GB+**: 可以使用 large 配置

### Q: 多 GPU 训练?

**A:** 参考 [分布式训练指南](../production/distributed-training.md)

```python
# 简要示例
torch.distributed.init_process_group(backend='nccl')
model = DDP(model, device_ids=[local_rank])
```

---

## 📚 相关文档

- [RTX 5060 完整指南](rtx-5060.md) - 8GB GPU 详细优化
- [Apple Silicon 指南](apple-silicon.md) - M 系列芯片优化
- [配置速查表](quick-reference.md) - 快速参考
- [分布式训练](../production/distributed-training.md) - 多 GPU 训练
- [混合精度](../production/mixed-precision.md) - FP16/BF16 优化

---

## 🎯 选择建议

### 预算导向

- **< ¥3000**: RTX 5060 (8GB) - 性价比之选
- **¥3000-10000**: RTX 4070 Ti / 4080 - 平衡之选
- **> ¥10000**: RTX 4090 - 性能之选
- **Mac 用户**: M4 Pro / Max - 生态之选

### 使用场景导向

- **学习 LLM**: CPU 或 RTX 5060
- **个人研究**: RTX 5060 / M4 Pro
- **专业开发**: RTX 4090
- **生产训练**: A100 云租用

---

**根据您的硬件，选择最适合的配置，开始高效训练！** 🚀
