# LLM Foundry

> **实用的开源 LLM 基础 —— 从基础到生产**

一个教育与生产并重的 Transformer 语言模型实现，涵盖从基础学习到生产部署的完整场景。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

[English](#) | [中文](#)

---

## 🎯 这是什么？

LLM Foundry 是一个**教育优先、生产就绪**的语言模型实现，采用**双轨设计**:

- **教学轨 (tutorials/)**: 单文件完整实现，详细注释，适合学习
- **生产轨 (src/)**: 模块化包，工程优化，适合开发

**核心特性**: RoPE • GQA • SwiGLU • RMSNorm • 现代 Transformer 架构

---

## 🚀 快速导航

### 首次使用

**5 分钟快速体验** → [GETTING_STARTED.md](GETTING_STARTED.md)
   - 安装 → 训练第一个模型 → 生成文本

**10-15 小时系统学习** → [LEARNING_PATH.md](LEARNING_PATH.md)
   - 5 阶段结构化路径
   - 从零理解 Transformer
   - 掌握核心技术

---

### 实践者

**针对您的硬件优化** → [硬件指南](docs/zh/hardware/)
   - [RTX 5060 指南](docs/zh/hardware/rtx-5060.md) - 8GB GPU (70M 参数, 30-40 分钟)
   - [Apple Silicon 指南](docs/zh/hardware/apple-silicon.md) - M4 Pro 优化
   - [配置速查表](docs/zh/hardware/quick-reference.md) - 快速参考

---

### 开发者 / 研究者

**深入架构** → [架构文档](docs/zh/architecture/)
   - [核心组件](docs/zh/architecture/components.md) - RMSNorm, RoPE, GQA, SwiGLU
   - [训练系统](docs/zh/architecture/training-system.md) - LLM 训练完整知识
   - [设计决策](docs/zh/architecture/design-decisions.md) - 技术选型理由

**贡献代码** → [AGENTS.md](AGENTS.md)
   - AI Agent 协作指南
   - 开发工作流
   - 贡献规范

---

## ⚡ 一分钟体验

```bash
# 1. 安装
git clone https://github.com/your-org/llm-foundry.git
cd llm-foundry
pip install -e .

# 2. 训练
cd tutorials
python train.py      # 训练模型 (~30 秒)

# 3. 生成
python generate.py   # 生成文本
```

**详细步骤** → [GETTING_STARTED.md](GETTING_STARTED.md)

---

## 📚 文档

**主导航** → [docs/README.md](docs/README.md)
   - 按用途导航 ("我想...")
   - 按角色导航 (初学者、实践者、开发者)
   - 完整文档索引

**核心文档**:
- [📖 快速开始](GETTING_STARTED.md) - 5-10 分钟上手
- [🎯 学习路径](LEARNING_PATH.md) - 系统学习指南
- [🏗️ 架构详解](docs/zh/architecture/) - 深入理解
- [💻 硬件指南](docs/zh/hardware/) - 平台优化
- [🚀 生产部署](docs/zh/production/) - 企业级部署
- [🤖 AI Agent](AGENTS.md) - 开发协作

---

## 🏗️ 项目结构

```
llm-foundry/
├── src/llm_foundry/      # 主包 (生产代码)
│   ├── models/           # Transformer 实现
│   ├── training/         # 训练工具
│   ├── inference/        # 推理工具
│   └── ...
├── tutorials/            # 教学脚本 (镜像 src/ 功能)
│   ├── model.py          # 完整 Transformer (单文件)
│   ├── train.py          # 训练流程
│   └── ...
├── docs/                 # 文档
│   └── zh/               # 中文文档
│       ├── architecture/ # 架构详解
│       ├── guides/       # 实用指南
│       ├── hardware/     # 硬件指南
│       └── production/   # 生产部署
├── examples/             # 使用示例
└── tests/                # 测试
```

---

## ✨ 特性

- 🎯 **现代架构**: RoPE, GQA, SwiGLU, RMSNorm 等最新技术
- 📦 **模块化设计**: 清晰的代码结构，易于理解和扩展
- 🎓 **教育友好**: 详细的中文文档和注释
- 🚀 **生产就绪**: 分布式训练、混合精度、模型服务
- 🔧 **双轨并行**: tutorials/ (教学) + src/ (生产)，功能对等

---

## 📊 模型配置

| 配置 | 参数量 | 层数 | 维度 | 适用场景 |
|------|--------|------|------|---------|
| Small | ~2M | 4 | 256 | 学习、CPU 训练 |
| Medium | ~10M | 8 | 512 | 实验、小 GPU |
| RTX 5060 | ~70M | 10 | 704 | 8GB GPU |
| Large | ~200M | 24 | 1024 | 高端 GPU/云 |

---

## 🎯 使用场景

### 教育学习
- 理解 Transformer 架构
- 学习 LLM 训练流程
- 实验模型设计

### 研究开发
- 快速原型验证
- 架构改进实验
- 算法优化测试

### 生产部署
- 定制化 LLM 解决方案
- 垂直领域模型训练
- 企业级部署

---

## 🛠️ 技术栈

- **框架**: PyTorch 2.0+
- **架构**: Decoder-Only Transformer
- **分词**: SentencePiece BPE
- **训练**: AdamW, 混合精度, DDP/FSDP
- **推理**: Top-k/Top-p 采样, KV Cache

---

## 💡 示例

查看 [examples/](examples/) 目录:

- `01_basic_training.py` - 基础训练
- `02_custom_data.py` - 自定义数据集
- `03_generation_sampling.py` - 采样策略
- `04_fine_tuning.py` - 模型微调

---

## 🤝 贡献

我们欢迎各种形式的贡献！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

详见 [AGENTS.md](AGENTS.md) 了解开发工作流。

---

## 📜 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

---

## 🌟 致谢

感谢以下项目的启发:

- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- [LLaMA](https://github.com/facebookresearch/llama) by Meta AI
- [Transformer](https://arxiv.org/abs/1706.03762) paper by Vaswani et al.

---

## 📞 联系方式

- 问题反馈: [GitHub Issues](https://github.com/your-org/llm-foundry/issues)
- 讨论交流: [GitHub Discussions](https://github.com/your-org/llm-foundry/discussions)

---

⭐ **如果这个项目对你有帮助，请给我们一个 Star！**
