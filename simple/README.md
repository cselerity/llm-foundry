# 简单模式 - 快速入门

这个目录包含简化的单文件脚本,用于快速体验和教学。

## 🎯 适用场景

- 快速实验和原型验证
- 学习 LLM 基础概念
- 教学演示
- 不需要完整包功能的简单任务

## 📁 文件说明

- `config.py` - 模型和训练配置
- `model.py` - 完整的模型实现
- `tokenizer.py` - 分词器
- `data.py` - 数据加载
- `train.py` - 训练脚本
- `generate.py` - 生成脚本

## 🚀 快速开始

### 1. 训练模型

```bash
cd simple
python train.py
```

这会:
- 自动下载红楼梦数据集
- 训练一个 SentencePiece 分词器
- 训练 Mini LLM 模型(默认 100 步)
- 保存模型为 `minillm.pt`

### 2. 生成文本

```bash
python generate.py
```

这会:
- 加载训练好的模型
- 使用提示词"满纸荒唐言,"生成文本
- 输出生成的结果

## ⚙️ 自定义配置

编辑 `config.py` 修改模型参数:

```python
@dataclass
class ModelConfig:
    dim: int = 256          # 增加以获得更大模型
    n_layers: int = 4       # 增加层数
    n_heads: int = 8        # 注意力头数
    # ...

@dataclass
class TrainConfig:
    max_iters: int = 1000   # 增加训练步数
    batch_size: int = 32    # 调整批次大小
    # ...
```

## 🆚 简单模式 vs 包模式

| 特性 | 简单模式 | 包模式 |
|------|----------|--------|
| 易用性 | ⭐⭐⭐⭐⭐ 极简 | ⭐⭐⭐ 需要理解模块 |
| 功能性 | ⭐⭐ 基础功能 | ⭐⭐⭐⭐⭐ 完整功能 |
| 扩展性 | ⭐⭐ 有限 | ⭐⭐⭐⭐⭐ 高度模块化 |
| 生产就绪 | ❌ 不推荐 | ✅ 推荐 |
| 学习曲线 | 平缓 | 中等 |

## 🎓 学习建议

1. **从简单模式开始**: 运行 `train.py` 和 `generate.py`,理解基本流程
2. **阅读代码**: 查看 `model.py`,理解 Transformer 架构
3. **实验参数**: 修改 `config.py`,观察效果
4. **过渡到包模式**: 准备好后,探索 `src/llm_foundry/` 的模块化实现

## 📚 下一步

准备好升级?查看:

- [完整文档](../docs/README.md)
- [包模式使用](../docs/zh/quickstart.md)
- [示例代码](../examples/)
- [Agent 协作指南](../AGENTS.md)

## ⚠️ 注意事项

- 简单模式适合学习和快速实验
- 生产环境请使用包模式 (`src/llm_foundry/`)
- 简单模式的功能更新可能不如包模式及时

---

祝学习愉快! 🎉
