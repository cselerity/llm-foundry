# LLM Foundry 文档中心

> **按用途和角色导航的完整文档索引**

欢迎来到 LLM Foundry 文档中心。本页面帮助您快速找到所需的文档。

---

## 🎯 我想...

### 快速开始 (5-10 分钟)

**→ [快速开始指南](../GETTING_STARTED.md)**
   - 安装环境
   - 训练第一个模型
   - 生成文本
   - 故障排除

---

### 系统学习 (10-15 小时)

**→ [学习路径 (LEARNING_PATH.md)](../LEARNING_PATH.md)**
   - 🎯 第 1 阶段: 快速入门 (30 分钟)
   - 🔧 第 2 阶段: 核心概念 (2-3 小时)
   - 🧠 第 3 阶段: 深入理解 (3-4 小时)
   - 🚀 第 4 阶段: 优化与扩展 (3-4 小时)
   - 💼 第 5 阶段: 生产部署 (2-3 小时)

---

### 理解架构

**→ [架构文档](zh/architecture/)**

**深入学习顺序:**
1. [**核心组件**](zh/architecture/components.md) - RMSNorm, RoPE, GQA, SwiGLU
2. [**训练系统**](zh/architecture/training-system.md) - LLM 训练完整知识
3. **设计决策** - 技术选型理由 (待创建)

---

### 在我的硬件上训练

**→ [硬件指南](zh/hardware/)**

| 硬件类型 | 指南 | 模型规模 | 训练时间* |
|---------|------|---------|----------|
| RTX 5060 (8GB) | [RTX 5060 指南](zh/hardware/rtx-5060.md) | 70M | 30-40min |
| Apple M4 Pro | Apple Silicon 指南 (待创建) | 68M | 40-60min |
| 高端 GPU | [硬件概览](zh/hardware/README.md) | 200M+ | 10-20min |

*基于 10k training steps

**→ [配置速查表](zh/hardware/quick-reference.md)** - 快速参考

---

### 使用自己的数据

**→ 自定义数据指南** (待创建)

**临时参考 - [GETTING_STARTED.md](../GETTING_STARTED.md#使用自己的数据)**

---

### 部署到生产

**→ [生产部署指南](zh/production/)**

- [**分布式训练**](zh/production/distributed-training.md) - 多 GPU 训练
- [**混合精度**](zh/production/mixed-precision.md) - FP16/BF16 加速
- [**模型服务**](zh/production/model-serving.md) - API 部署
- [**推理优化**](zh/production/optimization.md) - 量化和加速

---

### 贡献代码

**→ [开发者指南 (AGENTS.md)](../AGENTS.md)**
   - AI Agent 协作指南
   - 开发工作流
   - 代码规范

---

## 📚 按文档类型浏览

### 指南 (实用操作)

- [**快速开始**](../GETTING_STARTED.md) - 5-10 分钟上手 ✅
- [快速入门](zh/quickstart.md) - 原版快速入门 (已被 GETTING_STARTED.md 替代)

---

### 架构 (技术深入)

- [**核心组件**](zh/architecture/components.md) - RMSNorm, RoPE, GQA, SwiGLU ✅
- [**训练系统**](zh/architecture/training-system.md) - 完整训练知识 ✅
- 设计决策 - 技术选型 (待创建)

---

### 硬件 (平台特定)

- [**硬件概览**](zh/hardware/README.md) - 选择指南 ✅
- [**RTX 5060**](zh/hardware/rtx-5060.md) - 8GB NVIDIA GPU ✅
- [**配置速查表**](zh/hardware/quick-reference.md) - 快速参考 ✅

---

### 生产 (部署)

- [**分布式训练**](zh/production/distributed-training.md) - DDP, FSDP
- [**混合精度**](zh/production/mixed-precision.md) - FP16/BF16
- [**模型服务**](zh/production/model-serving.md) - FastAPI 部署
- [**推理优化**](zh/production/optimization.md) - 量化、KV Cache

---

## 📖 按角色学习路径

### 🎓 初学者

**推荐路径:**
1. [快速开始](../GETTING_STARTED.md) (5-10 分钟)
2. [学习路径 第 1-2 阶段](../LEARNING_PATH.md) (2-3 小时)
3. [核心组件](zh/architecture/components.md) (1-2 小时)

---

### 💻 实践者

**推荐路径:**
1. [快速开始](../GETTING_STARTED.md) (10 分钟)
2. [硬件指南](zh/hardware/) - 选择配置 (15 分钟)
3. [配置速查表](zh/hardware/quick-reference.md) (5 分钟)

---

### 🔬 研究者 / 开发者

**推荐路径:**
1. [完整学习路径](../LEARNING_PATH.md) (10-15 小时)
2. [架构文档](zh/architecture/) - 完整阅读
3. [开发者指南](../AGENTS.md) (1 小时)

---

## 🔗 快速链接

- [GitHub 仓库](https://github.com/your-org/llm-foundry)
- [Issue 追踪](https://github.com/your-org/llm-foundry/issues)
- [讨论区](https://github.com/your-org/llm-foundry/discussions)

---

**找到您需要的文档了吗？** 如果没有，请 [告诉我们](https://github.com/your-org/llm-foundry/issues) 📝
