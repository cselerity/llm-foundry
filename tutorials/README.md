# Tutorials - 教学教程

> **定位**: 这不是简化版本,而是主工程核心功能的**教学展示**

本目录提供独立运行的教程脚本,帮助您通过完整的、可运行的代码理解 LLM Foundry 的核心概念和工作流程。

## 📖 新手指南

**第一次学习?** 建议先查看完整的学习路径:

👉 **[学习路径指南 (../LEARNING_PATH.md)](../LEARNING_PATH.md)**

学习路径将告诉您:
- 应该按什么顺序阅读代码
- 每个文件的核心概念是什么
- 如何通过实践加深理解

## 🎯 教程定位

这些教程脚本是 `src/llm_foundry/` 主工程的**教学镜像**:

```
src/llm_foundry/           tutorials/
(工程化实现)          →   (教学展示)
├── models/            →   model.py (完整展示)
├── training/          →   train.py (流程展示)
├── inference/         →   generate.py (用法展示)
└── ...
```

**关键特点**:
- ✅ **功能完整**: 与主工程功能对等,不是简化版
- ✅ **独立运行**: 单文件即可运行,便于学习
- ✅ **详细注释**: 代码中包含丰富的教学注释
- ✅ **教学优先**: 代码结构优化为教学友好

## 📚 教程文件

| 文件 | 说明 | 对应主工程模块 |
|------|------|----------------|
| `config.py` | 配置类定义 | `src/llm_foundry/config/` |
| `model.py` | 完整的 Transformer 实现 | `src/llm_foundry/models/` |
| `tokenizer.py` | SentencePiece 分词器 | `src/llm_foundry/tokenizers/` |
| `data.py` | 数据加载和处理 | `src/llm_foundry/data/` |
| `train.py` | 训练流程 | `src/llm_foundry/training/` |
| `generate.py` | 文本生成 | `src/llm_foundry/inference/` |

## 🚀 快速开始

```bash
cd tutorials
python train.py      # 训练模型
python generate.py   # 生成文本
```

## 🔄 与主工程的关系

**核心原则**: 教程是主工程的**展示窗口**,不是独立分支。

- 教程优化为教学友好,展开所有细节
- 工程版本封装成模块,生产就绪
- 功能完全对等,只是组织方式不同

---

详细说明请查看完整文档。
