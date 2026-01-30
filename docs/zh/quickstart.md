# 快速入门

欢迎使用 LLM Foundry! 本指南将帮助您在 5 分钟内开始训练和使用您的第一个语言模型。

## 📋 前置要求

- Python 3.8 或更高版本
- 至少 4GB RAM
- (可选) NVIDIA GPU 用于加速训练

## 🚀 安装

### 方法 1: 从源码安装 (推荐)

```bash
# 克隆仓库
git clone https://github.com/your-org/llm-foundry.git
cd llm-foundry

# 安装依赖
pip install -e .

# (可选) 安装开发工具
pip install -r requirements-dev.txt
```

### 方法 2: 使用 pip 安装

```bash
pip install llm-foundry
```

## 🎯 两种使用模式

LLM Foundry 提供两种使用模式,适应不同的需求:

### 简单模式 - 快速体验 🎓

适合: 学习、教学、快速实验

```bash
cd simple
python train.py      # 训练模型
python generate.py   # 生成文本
```

### 包模式 - 生产使用 🏭

适合: 研究、生产、定制化开发

```python
from llm_foundry import ModelConfig, MiniLLM, DataLoader
```

本指南主要介绍**简单模式**,包模式请参考 [API 参考](api-reference.md)。

---

## 🎓 简单模式快速开始

### 步骤 1: 训练您的第一个模型

```bash
cd simple
python train.py
```

**发生了什么?**

1. 自动下载红楼梦数据集 (~100KB)
2. 训练 SentencePiece 分词器 (词表大小: 8192)
3. 训练 Mini LLM 模型 (默认 100 步)
4. 保存模型检查点到 `minillm.pt`

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
训练完成,耗时 32.45s
模型已保存至 minillm.pt
```

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

### 步骤 3: 自定义配置

编辑 `simple/config.py`:

```python
@dataclass
class ModelConfig:
    dim: int = 512          # 增大模型维度
    n_layers: int = 8       # 增加层数
    n_heads: int = 8
    vocab_size: int = 8192
    max_seq_len: int = 512  # 增加上下文长度

@dataclass
class TrainConfig:
    batch_size: int = 16
    learning_rate: float = 3e-4
    max_iters: int = 5000   # 训练更多步数
```

**重新训练:**

```bash
python train.py
```

---

## 🏭 包模式快速开始

### 步骤 1: 基本使用

```python
from llm_foundry import ModelConfig, MiniLLM, Tokenizer, DataLoader
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
print(f"使用设备: {device}")
```

### 步骤 2: 加载数据

```python
from llm_foundry import DataLoader

# 加载数据
loader = DataLoader(
    file_path='data/your_text.txt',  # 您的数据
    batch_size=32,
    block_size=cfg.max_seq_len,
    device=device
)

# 获取一个批次
x, y = loader.get_batch('train')
print(f"输入形状: {x.shape}")
print(f"目标形状: {y.shape}")
```

### 步骤 3: 训练模型

```python
import torch

# 创建优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# 训练循环
model.train()
for step in range(1000):
    # 获取批次
    x, y = loader.get_batch('train')

    # 前向传播
    logits, loss = model(x, y)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}: loss {loss.item():.4f}")

# 保存模型
torch.save(model.state_dict(), 'model.pt')
```

### 步骤 4: 生成文本

```python
from llm_foundry.inference import generate

# 加载模型
model.load_state_dict(torch.load('model.pt'))
model.eval()

# 准备提示词
tokenizer = loader.tokenizer
prompt = "满纸荒唐言，"
input_ids = torch.tensor(
    tokenizer.encode(prompt),
    dtype=torch.long,
    device=device
)[None, ...]

# 生成
with torch.no_grad():
    output_ids = generate(
        model,
        input_ids,
        max_new_tokens=100,
        temperature=0.8,
        top_k=50
    )

# 解码
generated_text = tokenizer.decode(output_ids[0].tolist())
print(generated_text)
```

---

## 📊 验证安装

运行以下命令验证安装:

```python
import llm_foundry
print(f"LLM Foundry 版本: {llm_foundry.__version__}")

# 测试导入
from llm_foundry import ModelConfig, MiniLLM, Tokenizer
print("✅ 所有模块导入成功!")
```

---

## 🎯 下一步

现在您已经成功运行了第一个模型,接下来可以:

### 深入学习
- 📖 阅读 [架构详解](architecture.md) 理解模型原理
- 🎓 查看 [训练指南](training.md) 学习训练技巧
- 🔍 探索 [推理指南](inference.md) 了解生成策略

### 实践项目
- 📝 使用自己的数据训练模型 → [数据准备](data-preparation.md)
- ⚙️ 自定义模型配置 → [配置系统](configuration.md)
- 🚀 部署模型服务 → [模型服务](production/model-serving.md)

### 高级主题
- 🔥 [分布式训练](production/distributed-training.md) - 多 GPU 训练
- ⚡ [混合精度](production/mixed-precision.md) - 加速训练
- 📈 [推理优化](production/optimization.md) - 提升推理速度

---

## ❓ 常见问题

### Q: 训练太慢怎么办?

**A:** 尝试以下方法:
- 使用 GPU: 确保安装了 CUDA 版本的 PyTorch
- 减小模型: 降低 `dim` 或 `n_layers`
- 减小批次: 降低 `batch_size`
- 使用混合精度: 参考 [混合精度训练](production/mixed-precision.md)

### Q: 显存不足 (OOM) 怎么办?

**A:**
- 减小 `batch_size`
- 减小 `max_seq_len`
- 减小模型大小 (`dim`, `n_layers`)
- 使用梯度累积

### Q: 生成的文本质量不好?

**A:**
- 增加训练步数 (`max_iters`)
- 使用更大的模型
- 使用更多训练数据
- 调整采样参数 (`temperature`, `top_k`, `top_p`)

### Q: 如何使用自己的数据?

**A:** 参考 [数据准备指南](data-preparation.md),简要步骤:
1. 准备纯文本文件
2. 使用 `DataLoader(file_path='your_data.txt')`
3. 训练分词器和模型

### Q: 简单模式和包模式有什么区别?

**A:**
- **简单模式**: 单文件脚本,适合快速实验和学习
- **包模式**: 模块化架构,适合生产和定制开发
- 功能相同,只是组织方式不同

---

## 🆘 获取帮助

遇到问题?

- 📚 查看 [完整文档](../README.md)
- 💬 提交 [GitHub Issue](https://github.com/your-org/llm-foundry/issues)
- 🗨️ 参与 [GitHub Discussions](https://github.com/your-org/llm-foundry/discussions)

---

## 🎉 恭喜!

您已经成功完成快速入门!现在您可以:
- ✅ 训练自己的语言模型
- ✅ 生成文本
- ✅ 理解基本工作流程

继续探索 LLM Foundry 的更多功能吧! 🚀
