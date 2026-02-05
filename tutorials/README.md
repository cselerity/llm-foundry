# Tutorials - 教学脚本

> **完整的、可运行的教学脚本，镜像 `src/llm_foundry/` 的功能**

---

## 🎯 这是什么？

这些教学脚本是 `src/llm_foundry/` 主工程的**教学镜像**:

- **不是简化版** - 与主工程功能完全对等
- **单文件实现** - 每个脚本独立运行，便于学习
- **详细注释** - 代码中包含丰富的教学注释
- **教学优先** - 代码结构优化为教学友好

```
src/llm_foundry/           tutorials/
(工程化实现)          →   (教学展示)
├── models/           →   model.py (完整展示)
├── training/         →   train.py (流程展示)
├── inference/        →   generate.py (用法展示)
├── config/           →   config.py (配置定义)
├── tokenizers/       →   tokenizer.py (分词器)
└── data/             →   dataloader.py (数据加载)

功能完全对等，只是组织方式不同
```

---

## 📚 教程文件

### 核心文件

| 文件 | 说明 | 行数 | 对应主工程模块 |
|------|------|------|----------------|
| [**config.py**](config.py) | 配置类定义 | ~700 | `src/llm_foundry/config/` |
| [**model.py**](model.py) | 完整 Transformer 实现 | ~630 | `src/llm_foundry/models/` |
| [**tokenizer.py**](tokenizer.py) | SentencePiece 分词器 | ~530 | `src/llm_foundry/tokenizers/` |
| [**dataloader.py**](dataloader.py) | 数据加载和处理 | ~430 | `src/llm_foundry/data/` |
| [**train.py**](train.py) | 训练流程 | ~420 | `src/llm_foundry/training/` |
| [**generate.py**](generate.py) | 文本生成 | ~480 | `src/llm_foundry/inference/` |

---

### 硬件优化脚本

| 文件 | 说明 | 适用硬件 |
|------|------|----------|
| [**train_rtx5060.py**](train_rtx5060.py) | RTX 5060 优化训练 | NVIDIA GPU (8GB) |
| [**train_m4pro.py**](train_m4pro.py) | Apple M4 Pro 优化训练 | Apple Silicon (32GB) |

**详细配置指南:**
- RTX 5060 → [完整指南](../docs/hardware-rtx5060.md)
- Apple Silicon → 指南 (待创建)

---

## 🚀 快速开始

### 基础训练 (适合学习)

```bash
cd tutorials
python train.py      # 训练模型 (小型配置)
python generate.py   # 生成文本
```

**需要详细说明?** 查看 [快速开始指南](../GETTING_STARTED.md)

---

### RTX 5060 优化训练

```bash
python train_rtx5060.py  # 70M 参数, 30-40 分钟
```

**配置详情**: [RTX 5060 指南](../docs/hardware-rtx5060.md)

---

### Apple M4 Pro 优化训练

```bash
python train_m4pro.py    # 68M 参数, 40-60 分钟
```

---

## 📖 学习路径

**首次学习?** 建议先查看完整的学习路径:

👉 **[学习路径指南 (../LEARNING_PATH.md)](../LEARNING_PATH.md)**

学习路径将告诉您:
- 应该按什么顺序阅读代码
- 每个文件的核心概念是什么
- 如何通过实践加深理解

---

## 🎓 推荐阅读顺序

### 第 1 步: 理解配置

**阅读**: [config.py](config.py)

**重点理解**:
- `ModelConfig` - 模型参数配置
- `TrainConfig` - 训练参数配置
- 预设配置函数 (small, medium, RTX 5060, M4 Pro)

---

### 第 2 步: 理解数据处理

**阅读**: [tokenizer.py](tokenizer.py) → [dataloader.py](dataloader.py)

**重点理解**:
- SentencePiece BPE 分词
- 数据加载和批处理
- 训练/验证集分割

---

### 第 3 步: 理解模型架构

**阅读**: [model.py](model.py)

**重点理解**:
- RMSNorm, RoPE, GQA, SwiGLU
- Transformer Block 结构
- 因果自注意力机制

**深入学习** → [架构组件详解](../docs/architecture-components.md)

---

### 第 4 步: 理解训练流程

**阅读**: [train.py](train.py)

**重点理解**:
- 训练循环
- 损失计算
- 梯度更新
- 检查点保存

---

### 第 5 步: 理解文本生成

**阅读**: [generate.py](generate.py)

**重点理解**:
- 自回归生成
- 采样策略 (temperature, top-k, top-p)
- 生成控制

---

## 🔄 与主工程的关系

### 核心原则

教程是主工程的**展示窗口**，不是独立分支。

### 使用场景

**教学脚本 (tutorials/) - 用于学习**:
- ✅ 单文件完整实现
- ✅ 详细注释
- ✅ 教学优先
- ✅ 独立运行

**主包 (src/) - 用于开发**:
- ✅ 模块化设计
- ✅ 简洁代码
- ✅ 性能优先
- ✅ 包管理

### 如何选择

```python
# 学习和理解 → 使用教学脚本
cd tutorials
python train.py

# 生产开发 → 使用主包
from llm_foundry import MiniLLM, Trainer, DataLoader
```

---

## 💻 硬件配置参考

### 推荐配置对比

| 用途 | GPU | 显存 | 模型规模 | 训练时间* |
|------|-----|------|---------|----------|
| **学习** | CPU | - | 2M | 10-30min |
| **实验** | RTX 3060 | 12GB | 10M | 5-10min |
| **本地训练** | **RTX 5060** | **8GB** | **70M** | **30-40min** |
| **专业开发** | RTX 4090 | 24GB | 200M+ | 10-20min |
| **Mac 用户** | **M4 Pro** | **32GB 统一** | **68M** | **40-60min** |

*基于 10k training steps

### 快速选择

```python
# RTX 5060 / 3060 / 4060 用户
from config import get_rtx5060_config, get_rtx5060_train_config
python train_rtx5060.py

# Apple M4 Pro / M3 Pro / M2 Pro 用户
from config import get_m4pro_config, get_m4pro_train_config
python train_m4pro.py

# 其他配置
from config import get_small_config, get_medium_config
python train.py
```

**详细配置指南**:
- [RTX 5060 指南](../docs/hardware-rtx5060.md) - 8GB GPU 优化
- [配置速查表](../docs/hardware-config.md) - 快速参考

---

## 🎓 深入学习

想要系统性学习 LLM 训练的完整知识？

### 架构深入

→ **[架构文档](../docs/)**
   - [核心组件](../docs/architecture-components.md) - RMSNorm, RoPE, GQA, SwiGLU
   - [训练系统](../docs/architecture-training.md) - LLM 训练完整知识体系

**训练系统文档包含**:
- 训练全流程 (6 阶段)
- 核心技术详解 (数据准备、预训练、SFT、RLHF)
- 业界最佳实践 (OpenAI, Meta, Anthropic, Google)
- 训练优化技巧 (速度、内存、质量)

---

## 📖 下一步

**完成教程学习后**:

1. **系统学习** → [学习路径](../LEARNING_PATH.md) - 结构化学习
2. **深入架构** → [架构文档](../docs/) - 技术深入
3. **硬件优化** → [硬件配置](../docs/) - 平台特定
4. **生产部署** → [生产文档](../docs/zh/production/) - 企业级
5. **贡献代码** → [开发者指南](../AGENTS.md) - 参与开发

---

## 🤝 贡献

发现问题或有改进建议？欢迎:
- 提交 Issue
- 创建 Pull Request
- 分享您的学习经验

详见 [AGENTS.md](../AGENTS.md) 了解贡献流程。

---

**祝学习愉快！🚀**

*通过教学脚本深入理解 LLM 实现，从基础到精通。*
