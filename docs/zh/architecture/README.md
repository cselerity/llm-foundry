# 架构文档

> **LLM Foundry 架构深入解析**

本目录包含 LLM Foundry 的完整架构文档，帮助您深入理解现代 Transformer 语言模型的实现。

---

## 📖 推荐阅读顺序

### 第 1 步: 核心组件详解

**→ [components.md](components.md)**

**阅读时间**: 1-2 小时

**内容概览**:
- Token Embedding - 词嵌入
- RMSNorm - 均方根归一化
- RoPE - 旋转位置编码
- Grouped Query Attention (GQA) - 分组查询注意力
- Scaled Dot-Product Attention - 缩放点积注意力
- MLP with SwiGLU - 门控前馈网络
- Transformer Block - 完整的 Transformer 层

**为什么先读这个？**
- 理解每个技术组件的数学原理
- 了解为什么选择这些现代技术
- 建立扎实的架构基础

---

### 第 2 步: 训练系统完整知识

**→ [training-system.md](training-system.md)**

**阅读时间**: 2-3 小时

**内容概览**:
- 训练全流程 (6 阶段)
  - 数据准备
  - 预训练
  - 监督微调 (SFT)
  - 奖励建模
  - RLHF
  - 评估与部署
- 核心技术详解
- 业界最佳实践 (OpenAI, Meta, Anthropic, Google)
- 训练优化技巧 (速度、内存、质量)

**为什么第二步读这个？**
- 理解完整的 LLM 训练流程
- 学习工业界的最佳实践
- 掌握训练优化技巧

---

### 第 3 步: 设计决策 (待创建)

**→ design-decisions.md** (计划中)

**内容概览**:
- 为什么选择 Decoder-Only 架构？
- 为什么使用 RoPE 而不是绝对位置编码？
- 为什么使用 GQA 而不是 MHA？
- 为什么使用 SwiGLU 而不是 ReLU？
- 为什么使用 Pre-Normalization？

---

## 🎯 按需求阅读

### 我想快速了解架构

**快速路径**:
1. 阅读 [components.md](components.md) 的前 3 个组件 (30 分钟)
   - RMSNorm
   - RoPE
   - GQA
2. 查看 Transformer Block 的完整流程图 (10 分钟)

**总时间**: ~40 分钟

---

### 我想深入理解每个技术细节

**深入路径**:
1. [components.md](components.md) - 完整阅读，理解每个公式 (2 小时)
2. 查看代码实现:
   - [src/llm_foundry/models/components.py](../../../src/llm_foundry/models/components.py)
   - [tutorials/model.py](../../../tutorials/model.py)
3. [training-system.md](training-system.md) - 完整阅读 (2-3 小时)
4. 动手实践 - 修改配置，观察效果 (1-2 小时)

**总时间**: 5-7 小时

---

### 我想了解训练优化

**优化路径**:
1. [training-system.md](training-system.md) - 第 4 节: 训练优化技巧 (1 小时)
   - 提升训练速度
   - 减少内存占用
   - 提升模型质量
   - 避免常见问题
2. [硬件指南](../hardware/) - 硬件特定优化 (30 分钟)
3. [生产文档](../production/) - 分布式训练、混合精度 (1 小时)

**总时间**: 2.5 小时

---

## 🔗 相关文档

### 代码导航

**主工程实现**:
- [src/llm_foundry/models/transformer.py](../../../src/llm_foundry/models/transformer.py) - 完整模型
- [src/llm_foundry/models/components.py](../../../src/llm_foundry/models/components.py) - 组件实现
- [src/llm_foundry/training/trainer.py](../../../src/llm_foundry/training/trainer.py) - 训练器

**教学实现**:
- [tutorials/model.py](../../../tutorials/model.py) - 教学版模型 (详细注释)
- [tutorials/train.py](../../../tutorials/train.py) - 教学版训练

**测试用例**:
- [tests/test_models.py](../../../tests/test_models.py) - 单元测试

---

### 学习路径

- **初学者** → [学习路径 第 3 阶段](../../../LEARNING_PATH.md#第三阶段深入理解)
- **实践者** → [硬件指南](../hardware/)
- **研究者** → 完整阅读本目录所有文档

---

### 其他架构资源

**外部参考**:
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
- [RoFormer](https://arxiv.org/abs/2104.09864) - RoPE 论文
- [RMSNorm](https://arxiv.org/abs/1910.07467) - RMSNorm 论文
- [LLaMA](https://arxiv.org/abs/2302.13971) - GQA, SwiGLU 应用
- [GQA](https://arxiv.org/abs/2305.13245) - GQA 论文

---

## 📊 文档结构

```
docs/zh/architecture/
├── README.md              # 本文件 - 阅读指南
├── components.md          # ✅ 核心组件详解 (必读)
├── training-system.md     # ✅ LLM 训练完整知识 (必读)
└── design-decisions.md    # ⏳ 设计决策 (计划中)
```

---

## 💡 学习建议

### 1. 循序渐进
- 不要跳过基础组件直接学训练
- 先理解"是什么"，再理解"为什么"

### 2. 动手实践
- 边读文档边看代码
- 修改配置参数观察效果
- 运行 tutorials/ 中的教学脚本

### 3. 对比学习
- 将本项目与其他实现对比（如 nanoGPT）
- 理解不同设计选择的权衡

### 4. 提出问题
- 遇到不理解的地方记录下来
- 在 [GitHub Discussions](https://github.com/your-org/llm-foundry/discussions) 提问

---

## 🎯 检查清单

阅读完架构文档后，您应该能够：

- [ ] 解释 RMSNorm 相比 LayerNorm 的优势
- [ ] 说明 RoPE 如何工作以及为什么有更好的外推能力
- [ ] 理解 GQA 如何在性能和质量间取得平衡
- [ ] 描述 Transformer Block 的完整数据流
- [ ] 说明 LLM 训练的 6 个阶段
- [ ] 列举至少 3 种训练优化技巧
- [ ] 解释主要的设计决策理由

---

**准备好深入理解 LLM 架构了吗？从 [components.md](components.md) 开始吧！** 🚀
