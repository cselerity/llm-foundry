# LLM Foundry

> **实用的开源 LLM 基础 —— 从基础到生产**

一个轻量级、模块化的 Transformer 语言模型实现,涵盖从基础概念到生产部署的完整旅程。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

[English](#) | [中文](#)

## ✨ 特性

- 🎯 **现代架构**: RoPE, GQA, SwiGLU, RMSNorm 等最新技术
- 📦 **模块化设计**: 清晰的代码结构,易于理解和扩展
- 🎓 **教育友好**: 详细的中文文档和注释
- 🚀 **生产就绪**: 支持分布式训练、混合精度、模型服务
- 🔧 **双模式**: 简单脚本(教学)+ 完整包(生产)

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/your-org/llm-foundry.git
cd llm-foundry

# 安装依赖
pip install -e .
```

### 简单模式(快速体验)

```bash
# 训练模型(使用简单脚本)
cd simple
python train.py

# 生成文本
python generate.py
```

### 包模式(生产使用)

```python
from llm_foundry import ModelConfig, MiniLLM, Tokenizer, DataLoader

# 1. 配置
cfg = ModelConfig()

# 2. 加载数据
loader = DataLoader(batch_size=32, block_size=256)

# 3. 创建模型
model = MiniLLM(cfg)

# 4. 训练
# ... (查看 docs/zh/training.md)
```

## 📖 文档

完整文档请访问 **[docs/](docs/README.md)**

### 核心文档

- **[快速入门](docs/zh/quickstart.md)** - 5分钟上手
- **[架构详解](docs/zh/architecture.md)** - 深入理解模型
- **[训练指南](docs/zh/training.md)** - 训练技巧和优化
- **[Agent 协作指南](AGENTS.md)** - AI Agent 开发指南

### 生产部署

- [分布式训练](docs/zh/production/distributed-training.md) - 多 GPU 训练
- [混合精度](docs/zh/production/mixed-precision.md) - FP16/BF16 加速
- [模型服务](docs/zh/production/model-serving.md) - API 部署
- [推理优化](docs/zh/production/optimization.md) - 量化和加速

## 🏗️ 项目结构

```
llm-foundry/
├── src/llm_foundry/      # 主包(生产代码)
│   ├── config/           # 配置
│   ├── models/           # 模型实现
│   ├── tokenizers/       # 分词器
│   ├── data/             # 数据处理
│   ├── training/         # 训练工具
│   ├── inference/        # 推理工具
│   └── utils/            # 工具函数
├── simple/               # 简单脚本(教学)
├── examples/             # 使用示例
├── docs/                 # 文档
├── tests/                # 测试
└── AGENTS.md             # Agent 协作指南
```

## 💡 示例

查看 [examples/](examples/) 目录获取更多示例:

- `01_basic_training.py` - 基础训练
- `02_custom_data.py` - 自定义数据集
- `03_generation_sampling.py` - 采样策略
- `04_fine_tuning.py` - 模型微调

## 🎯 使用场景

### 教育学习
- 理解 Transformer 架构
- 学习 LLM 训练流程
- 实验不同的模型设计

### 研究开发
- 快速原型验证
- 架构改进实验
- 算法优化测试

### 生产部署
- 定制化 LLM 解决方案
- 垂直领域模型训练
- 企业级部署

## 🛠️ 技术栈

- **框架**: PyTorch 2.0+
- **分词**: SentencePiece BPE
- **训练**: AdamW, 混合精度, DDP
- **推理**: Top-k/Top-p 采样, KV Cache

## 🤝 贡献

我们欢迎各种形式的贡献!

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

详见 [AGENTS.md](AGENTS.md) 了解开发工作流。

## 📊 模型参数

| 配置 | 参数量 | 层数 | 维度 | 头数 |
|------|--------|------|------|------|
| Small | ~2M | 4 | 256 | 8 |
| Medium | ~10M | 8 | 512 | 8 |
| Large | ~40M | 12 | 768 | 12 |

## 📜 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🌟 致谢

感谢以下项目的启发:

- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- [LLaMA](https://github.com/facebookresearch/llama) by Meta AI
- [Transformer](https://arxiv.org/abs/1706.03762) paper by Vaswani et al.

## 📞 联系方式

- 问题反馈: [GitHub Issues](https://github.com/your-org/llm-foundry/issues)
- 讨论交流: [GitHub Discussions](https://github.com/your-org/llm-foundry/discussions)

---

⭐ 如果这个项目对你有帮助,请给我们一个 Star!
