# LLM Foundry 快速开始

> **5-10 分钟上手，训练您的第一个语言模型**

欢迎使用 LLM Foundry！本指南将帮助您快速开始训练和使用您的第一个语言模型。

---

## 📋 前置要求

- **Python** 3.8 或更高版本
- **RAM** 至少 4GB
- **GPU** (可选) NVIDIA GPU 用于加速训练

### 检查环境

```bash
python --version  # 应该 >= 3.8
pip --version     # 确保 pip 已安装
```

---

## 🚀 安装

### 方法 1: 从源码安装 (推荐)

```bash
# 1. 克隆仓库
git clone https://github.com/your-org/llm-foundry.git
cd llm-foundry

# 2. 安装依赖
pip install -e .

# 3. (可选) 安装开发工具
pip install -r requirements-dev.txt
```

### 方法 2: 使用 pip 安装

```bash
pip install llm-foundry
```

### 验证安装

```python
import llm_foundry
print(f"LLM Foundry 版本: {llm_foundry.__version__}")

# 测试导入
from llm_foundry import ModelConfig, MiniLLM
print("✅ 所有模块导入成功!")
```

---

## 🎯 第一个模型 (5 分钟)

### 步骤 1: 训练模型

```bash
cd tutorials
python train.py
```

**正在发生什么?**

1. 📥 **下载数据**: 自动下载红楼梦数据集 (~100KB)
2. 🔤 **训练分词器**: 使用 SentencePiece BPE (词表大小: 8192)
3. 🧠 **训练模型**: 训练 Mini LLM 模型 (默认 100 步，~2M 参数)
4. 💾 **保存检查点**: 保存到 `minillm.pt`

**预期输出:**

```
使用设备: cuda
正在下载数据...
正在训练 Tokenizer (vocab_size=8192)...
Tokenizer 训练完成。
数据加载完成。总 token 数: 145234
模型参数量: 2.08M
开始训练...
step 0: train loss 9.1234, val loss 9.2345
step 50: train loss 5.6789, val loss 5.7890
训练完成，耗时 32.45s
模型已保存至 minillm.pt
```

---

### 步骤 2: 生成文本

```bash
python generate.py
```

**预期输出:**

```
使用设备: cuda
已加载 Checkpoint 'minillm.pt'

提示词: 满纸荒唐言，
正在生成...

--- 生成的文本 ---
满纸荒唐言，一把辛酸泪。都云作者痴，谁解其中味？...
```

---

### 步骤 3: 理解输出

**训练过程:**
- `train loss`: 训练集损失，应该逐步下降
- `val loss`: 验证集损失，评估模型泛化能力
- 损失从 ~9 降到 ~5-6 是正常的

**生成质量:**
- 初始模型 (100 步): 可能生成较短或重复的文本
- 需要更多训练步数 (1000-5000 步) 获得更好质量

---

## 🎓 两种使用模式

LLM Foundry 提供两种使用模式，适应不同的需求:

### 模式 1: 教学模式 (tutorials/)

**适合**: 学习、教学、快速实验

```bash
cd tutorials
python train.py      # 训练模型
python generate.py   # 生成文本
```

**特点:**
- ✅ 单文件完整实现
- ✅ 详细注释说明
- ✅ 独立运行
- ✅ 教学优先

---

### 模式 2: 包模式 (src/)

**适合**: 研究、生产、定制化开发

```python
from llm_foundry import ModelConfig, MiniLLM, DataLoader
from llm_foundry.utils import get_device

# 1. 配置
cfg = ModelConfig(
    dim=512,
    n_layers=8,
    n_heads=8,
    vocab_size=8192,
    max_seq_len=512
)

# 2. 创建模型
device = get_device()
model = MiniLLM(cfg).to(device)
print(f"模型参数量: {model.get_num_params()/1e6:.2f}M")

# 3. 加载数据
loader = DataLoader(
    file_path='data/your_text.txt',
    batch_size=32,
    block_size=cfg.max_seq_len,
    device=device
)

# 4. 训练 (简化示例)
import torch
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

model.train()
for step in range(1000):
    x, y = loader.get_batch('train')
    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}: loss {loss.item():.4f}")

# 5. 保存
torch.save(model.state_dict(), 'model.pt')
```

**特点:**
- ✅ 模块化 API
- ✅ 生产就绪
- ✅ 易于集成
- ✅ 性能优先

---

## 📊 自定义配置

### 方法 1: 编辑配置文件 (教学模式)

编辑 `tutorials/config.py`:

```python
@dataclass
class ModelConfig:
    dim: int = 512          # 增大模型维度
    n_layers: int = 8       # 增加层数
    n_heads: int = 8
    n_kv_heads: int = 4
    vocab_size: int = 8192
    max_seq_len: int = 512  # 增加上下文长度

@dataclass
class TrainConfig:
    batch_size: int = 16
    learning_rate: float = 3e-4
    max_iters: int = 5000   # 训练更多步数
    eval_interval: int = 100
```

**重新训练:**

```bash
cd tutorials
python train.py
```

---

### 方法 2: 使用预设配置

```python
# 小型模型 (适合学习, CPU)
from tutorials.config import get_small_config, get_small_train_config
model_cfg = get_small_config()      # ~2M 参数
train_cfg = get_small_train_config()

# 中型模型 (适合实验)
from tutorials.config import get_medium_config
model_cfg = get_medium_config()     # ~10M 参数

# RTX 5060 优化配置
from tutorials.config import get_rtx5060_config
model_cfg = get_rtx5060_config()    # ~70M 参数
```

---

## 🔧 根据用例选择下一步

### 💡 系统学习

想要深入理解 LLM 原理？

→ **[学习路径 (LEARNING_PATH.md)](LEARNING_PATH.md)**
   - 5 阶段结构化学习
   - 10-15 小时完整课程
   - 从基础到高级

→ **[架构详解](docs/)**
   - [核心组件](docs/architecture-components.md) - RMSNorm, RoPE, GQA
   - [训练系统](docs/architecture-training.md) - 完整训练知识
   - [设计决策](docs/architecture-design.md) - 技术选型

---

### 🖥️ 特定硬件优化

需要在特定硬件上优化？

→ **[硬件指南](docs/)**
   - [RTX 5060 指南](docs/hardware-rtx5060.md) - 8GB GPU 优化
   - [Apple Silicon 指南](docs/hardware-apple.md) - M4 Pro 优化
   - [配置速查表](docs/hardware-config.md) - 快速参考

**硬件选择参考:**

| 硬件 | 模型规模 | 训练时间* | 指南 |
|------|---------|----------|------|
| CPU | 2M | 10-30min | 使用 small 配置 |
| RTX 5060 (8GB) | 70M | 30-40min | [RTX 5060 指南](docs/hardware-rtx5060.md) |
| Apple M4 Pro | 68M | 40-60min | [Apple Silicon 指南](docs/hardware-apple.md) |
| RTX 4090 (24GB) | 200M+ | 10-20min | 自定义配置 |

*基于 10k training steps

---

### 📝 使用自己的数据

需要在自定义数据上训练？

→ **[自定义数据指南](docs/guides-data.md)** (待创建)

**快速步骤:**

```python
# 1. 准备纯文本文件
# your_data.txt

# 2. 使用教学模式
cd tutorials
# 编辑 train.py，修改 data_file 路径
python train.py

# 3. 或使用包模式
from llm_foundry import DataLoader

loader = DataLoader(
    file_path='your_data.txt',
    batch_size=32,
    block_size=256
)
```

---

### 🚀 生产部署

准备部署到生产环境？

→ **[生产部署](docs/)**
   - [分布式训练](docs/production-distributed.md) - 多 GPU 训练
   - [混合精度](docs/production-mixed.md) - FP16/BF16 加速
   - [模型服务](docs/production-serving.md) - API 部署
   - [推理优化](docs/production-optimize.md) - 量化和加速

---

## ❓ 故障排除

### 问题 1: 训练太慢

**症状:** 训练速度很慢，每步需要很长时间

**解决方案:**
1. **检查 GPU**:
   ```python
   import torch
   print(f"CUDA 可用: {torch.cuda.is_available()}")
   print(f"当前设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
   ```
2. **减小模型**: 降低 `dim` 或 `n_layers`
3. **减小批次**: 降低 `batch_size`
4. **使用混合精度**: 参考 [混合精度训练](docs/production-mixed.md)

---

### 问题 2: CUDA Out of Memory (OOM)

**症状:** 训练时出现显存不足错误

**解决方案:**
```python
# 方法 1: 减小 batch_size
train_cfg.batch_size = 16  # 或更小

# 方法 2: 减小 max_seq_len
model_cfg.max_seq_len = 128  # 或更小

# 方法 3: 减小模型大小
model_cfg.dim = 256
model_cfg.n_layers = 4

# 方法 4: 使用梯度累积
# 在 train.py 中实现
```

---

### 问题 3: 生成的文本质量不好

**症状:** 生成文本不连贯或重复

**解决方案:**
1. **增加训练步数**:
   ```python
   train_cfg.max_iters = 5000  # 而不是 100
   ```
2. **使用更大的模型**:
   ```python
   model_cfg = get_medium_config()  # 10M 参数
   ```
3. **使用更多训练数据**: 确保有足够的训练文本
4. **调整采样参数**:
   ```python
   # 在 generate.py 中
   temperature = 0.8   # 降低随机性
   top_k = 50          # 限制候选词
   top_p = 0.9         # 核采样
   ```

---

### 问题 4: 找不到模块

**症状:** `ModuleNotFoundError: No module named 'llm_foundry'`

**解决方案:**
```bash
# 确保在正确的目录
cd llm-foundry

# 重新安装
pip install -e .

# 验证安装
python -c "import llm_foundry; print('OK')"
```

---

## 💬 获取帮助

遇到其他问题？

- 📚 **查看完整文档**: [docs/README.md](docs/README.md)
- 🐛 **提交 Issue**: [GitHub Issues](https://github.com/your-org/llm-foundry/issues)
- 💬 **讨论**: [GitHub Discussions](https://github.com/your-org/llm-foundry/discussions)
- 📖 **学习路径**: [LEARNING_PATH.md](LEARNING_PATH.md)

---

## 🎉 恭喜！

您已经成功完成快速入门！现在您可以:
- ✅ 训练自己的语言模型
- ✅ 生成文本
- ✅ 理解基本工作流程
- ✅ 选择适合您的学习路径

### 推荐下一步

**如果您想...**

- **深入学习** → [LEARNING_PATH.md](LEARNING_PATH.md) (第 1 阶段)
- **理解架构** → [架构组件](docs/architecture-components.md)
- **优化硬件** → [硬件配置](docs/)
- **生产部署** → [生产部署](docs/)

---

**继续探索 LLM Foundry 的更多功能吧！** 🚀
