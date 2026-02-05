# LLM Foundry 学习路径

> **为初学者设计的循序渐进学习指南**

本指南为您提供一个结构化的学习路径,帮助您从零开始掌握 LLM Foundry 项目,理解 Transformer 架构,并最终能够进行生产级别的开发。

---

## 🎯 学习目标

完成本学习路径后,您将能够:

- ✅ 理解现代 Transformer 架构的核心原理
- ✅ 掌握 LLM 的训练和推理流程
- ✅ 能够自定义模型配置和训练参数
- ✅ 理解关键技术: RoPE, GQA, SwiGLU, RMSNorm
- ✅ 进行生产级别的模型开发和部署

---

## 📚 学习路径 (按顺序阅读)

### 🌟 第一阶段: 快速入门 (30分钟)

**目标**: 快速上手,运行第一个模型

#### 1. 项目概览
📖 **阅读**: [README.md](README.md)
- 了解项目定位和特性
- 理解双轨并行设计理念
- 查看项目结构

#### 2. 环境准备
📖 **阅读**: [GETTING_STARTED.md](GETTING_STARTED.md) - 安装部分
- 克隆仓库
- 安装依赖
- 验证环境

#### 3. 第一次训练
🎯 **实践**: 运行教学脚本
```bash
cd tutorials
python train.py      # 观察训练过程
python generate.py   # 体验文本生成
```

**关键点**:
- 观察训练损失的下降
- 尝试不同的生成提示词
- 理解基本的训练-生成流程

---

### 🔍 第二阶段: 理解代码 (2-3小时)

**目标**: 深入理解核心实现

#### 4. 配置系统
📖 **阅读**: [tutorials/config.py](tutorials/config.py)
- 理解 `ModelConfig`: 模型架构配置
- 理解 `TrainConfig`: 训练超参数配置
- 尝试修改参数,观察效果

**建议实践**:
```python
# 尝试修改这些参数,观察变化
cfg = ModelConfig(
    dim=128,           # 减小维度,加快训练
    n_layers=2,        # 减少层数
    vocab_size=4096    # 减小词表
)
```

#### 5. 模型架构 - 核心组件
📖 **按顺序阅读**: [src/llm_foundry/models/components.py](src/llm_foundry/models/components.py)

**阅读顺序**:

##### 5.1 RMSNorm (第18-64行)
- 🎓 **学习重点**: 归一化技术
- 📖 阅读类定义和 `forward` 方法
- 💡 **关键理解**: 为什么 RMSNorm 比 LayerNorm 更高效?

##### 5.2 RoPE 位置编码 (第66-148行)
- 🎓 **学习重点**: 位置信息编码
- 📖 阅读 `precompute_freqs_cis` 和 `apply_rotary_emb`
- 💡 **关键理解**:
  - RoPE 如何通过旋转编码位置信息?
  - 为什么相对位置在点积中自然体现?

##### 5.3 注意力机制 (第150-258行)
- 🎓 **学习重点**: Self-Attention 和 GQA
- 📖 阅读 `CausalSelfAttention` 类
- 💡 **关键理解**:
  - Q、K、V 的作用
  - GQA 如何减少参数和内存
  - 因果掩码的作用

**配套测试**: [tests/test_models.py](tests/test_models.py) - TestRoPE, TestCausalSelfAttention
- 运行测试理解组件行为
- 阅读测试中的"教学要点"

##### 5.4 前馈网络 (第260-278行)
- 🎓 **学习重点**: SwiGLU 激活函数
- 📖 阅读 `MLP` 类
- 💡 **关键理解**:
  - 门控机制的作用
  - 为什么 SwiGLU 比 ReLU 更好?

##### 5.5 Transformer 块 (第280-312行)
- 🎓 **学习重点**: Pre-normalization 架构
- 📖 阅读 `Block` 类
- 💡 **关键理解**:
  - Pre-norm vs Post-norm 的区别
  - 残差连接的作用

#### 6. 完整模型
📖 **阅读**: [src/llm_foundry/models/transformer.py](src/llm_foundry/models/transformer.py)
- 理解如何组装各个组件
- 理解 `forward` 方法的完整流程
- 查看参数量计算

**建议实践**:
```python
from llm_foundry import ModelConfig, MiniLLM

cfg = ModelConfig()
model = MiniLLM(cfg)

# 查看模型结构
print(model)

# 计算参数量
print(f"参数量: {model.get_num_params() / 1e6:.2f}M")
```

#### 7. 数据处理
📖 **阅读**: [tutorials/data.py](tutorials/data.py)
- 理解数据加载流程
- 理解 batch 的构造
- 理解因果语言建模的训练目标

#### 8. 分词器
📖 **阅读**: [tutorials/tokenizer.py](tutorials/tokenizer.py)
- 理解 BPE 分词原理
- 理解特殊 token (BOS, EOS, PAD)

---

### 📐 第三阶段: 深入理解 (3-4小时)

**目标**: 掌握核心原理和设计决策

#### 9. 架构深度解析
📖 **阅读**: [docs/architecture-components.md](docs/architecture-components.md)

**阅读重点**:
- 完整架构图解
- 各组件的设计决策
- 与其他模型的对比
- 参数量计算方法

#### 10. 训练流程详解
📖 **阅读**: [tutorials/train.py](tutorials/train.py) 和 [src/llm_foundry/training/trainer.py](src/llm_foundry/training/trainer.py)

**理解要点**:
- 训练循环的实现
- 损失计算和反向传播
- 优化器和学习率
- 验证和早停

**配套示例**: [examples/01_basic_training.py](examples/01_basic_training.py)

#### 11. 推理和生成
📖 **阅读**: [tutorials/generate.py](tutorials/generate.py) 和 [src/llm_foundry/inference/generator.py](src/llm_foundry/inference/generator.py)

**理解要点**:
- 自回归生成流程
- Temperature 的作用
- Top-k 和 Top-p 采样
- 生成质量控制

**配套示例**: [examples/03_generation_sampling.py](examples/03_generation_sampling.py)
- 运行不同采样策略
- 观察生成质量差异

#### 12. 运行测试
🎯 **实践**: 运行完整测试套件
```bash
# 安装测试依赖
pip install pytest pytest-cov

# 运行测试
pytest tests/ -v

# 查看覆盖率
pytest tests/ --cov=src/llm_foundry --cov-report=html
```

**学习建议**:
- 阅读每个测试的文档字符串
- 理解"教学要点"部分
- 尝试修改测试,加深理解

---

### 🚀 第四阶段: 实践应用 (4-6小时)

**目标**: 能够自定义和扩展

#### 13. 自定义数据集
📖 **阅读**: [tutorials/dataloader.py](tutorials/dataloader.py)
🎯 **实践**: [examples/02_custom_data.py](examples/02_custom_data.py)

**任务**:
- 准备自己的文本数据
- 训练自定义词表
- 在自己的数据上训练模型

#### 14. 调整模型配置
🎯 **实践**: 尝试不同的模型配置

```python
# 小型模型 (快速实验)
small_cfg = ModelConfig(
    dim=256, n_layers=4, n_heads=8, n_kv_heads=4
)

# 中型模型 (更好效果)
medium_cfg = ModelConfig(
    dim=512, n_layers=8, n_heads=8, n_kv_heads=4
)
```

**观察**:
- 参数量的变化
- 训练速度的差异
- 生成质量的提升

#### 15. 超参数调优
📖 **阅读**: [tutorials/train.py](tutorials/train.py)

**实验**:
- 调整学习率
- 调整 batch size
- 调整 warmup 步数
- 使用 learning rate scheduler

#### 16. 高级生成技巧
🎯 **深入实践**: [examples/03_generation_sampling.py](examples/03_generation_sampling.py)

**探索**:
- Temperature 对多样性的影响
- Top-k 和 Top-p 的平衡
- 组合使用多种策略

---

### 🏭 第五阶段: 生产实践 (可选,6-8小时)

**目标**: 掌握生产级别的技能

#### 17. 使用包模式开发
📖 **阅读**: [docs/README.md](docs/README.md)

**实践**:
```python
# 使用模块化 API
from llm_foundry import (
    ModelConfig, TrainConfig,
    MiniLLM, Tokenizer, DataLoader
)
from llm_foundry.training import Trainer
from llm_foundry.inference import Generator

# 构建完整应用
# ...
```

#### 18. 命令行工具
🎯 **实践**: 使用生产脚本

```bash
# 使用配置文件训练
python scripts/train.py --config configs/medium.yaml

# 生成文本
python scripts/generate.py \
    --checkpoint model.pt \
    --prompt "Once upon a time" \
    --temperature 0.8
```

#### 19. 生产部署 (高级)
📖 **阅读**: [docs/](docs/) 系列文档

- [分布式训练](docs/distributed-training.md)
- [混合精度训练](docs/mixed-precision.md)
- [模型服务](docs/model-serving.md)
- [推理优化](docs/optimization.md)

---

## 🎓 学习检查清单

完成以下检查项,确保您已掌握核心概念:

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

## 💡 学习建议

### 1. 循序渐进
- 严格按照路径顺序学习
- 不要跳过基础部分
- 每个阶段都要动手实践

### 2. 理解为先
- 不要只运行代码,要理解原理
- 阅读代码注释,理解设计决策
- 参考测试用例加深理解

### 3. 动手实践
- 每个概念都要写代码验证
- 尝试修改参数,观察效果
- 在自己的数据上实验

### 4. 记录笔记
- 记录重要概念和理解
- 记录遇到的问题和解决方法
- 总结关键知识点

### 5. 参考资源
- 代码注释是最好的教材
- 测试用例展示正确用法
- 文档提供系统性知识

---

## 🤝 获取帮助

遇到问题时:

1. **查看文档**: 优先查看相关文档
2. **阅读代码**: 代码注释包含详细解释
3. **运行测试**: 测试用例展示正确用法
4. **查看示例**: examples/ 目录包含完整示例
5. **提问讨论**: [GitHub Discussions](https://github.com/your-org/llm-foundry/discussions)
6. **报告问题**: [GitHub Issues](https://github.com/your-org/llm-foundry/issues)

---

## 📖 相关资源

### 项目文档
- [README.md](README.md) - 项目概览
- [AGENTS.md](AGENTS.md) - AI Agent 协作指南
- [docs/](docs/) - 完整文档系统

### 外部资源
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 原始 Transformer 论文
- [RoFormer](https://arxiv.org/abs/2104.09864) - RoPE 论文
- [LLaMA](https://arxiv.org/abs/2302.13971) - 现代 LLM 架构参考
- [GPT-3](https://arxiv.org/abs/2005.14165) - 大规模语言模型

---

## ⏱️ 预计学习时间

| 阶段 | 内容 | 时间 |
|------|------|------|
| 第一阶段 | 快速入门 | 30分钟 |
| 第二阶段 | 理解代码 | 2-3小时 |
| 第三阶段 | 深入理解 | 3-4小时 |
| 第四阶段 | 实践应用 | 4-6小时 |
| 第五阶段 | 生产实践 | 6-8小时(可选) |
| **总计** | | **10-15小时** |

*注: 时间仅供参考,实际学习时间因人而异*

---

## 🎉 完成后的下一步

恭喜您完成学习路径!现在您可以:

1. **深度探索**: 阅读 [AGENTS.md](AGENTS.md),学习如何贡献代码
2. **应用实践**: 在实际项目中应用所学知识
3. **分享经验**: 在社区中分享您的学习心得
4. **持续学习**: 关注最新的 LLM 研究进展

---

**祝您学习愉快!** 🚀

如有任何问题或建议,欢迎在 [GitHub Discussions](https://github.com/your-org/llm-foundry/discussions) 中讨论。
