# MiniLLM: 纯 PyTorch 实现的 LLM

这是一个用于教学目的的 LLM 项目，使用纯 PyTorch 从零构建了一个类似 Llama 的 Transformer 模型。

## 1. 环境准备

首先，确保你安装了 Python 3。然后安装项目依赖：

```bash
pip3 install -r requirements.txt
```

依赖项包括：
- `torch`: 深度学习框架
- `numpy`: 数值计算
- `sentencepiece`: Google 开源的分词器 (用于训练自定义 Tokenizer)

## 2. 数据准备

项目会自动下载《红楼梦》数据集 (`input_cn.txt`)。你不需要手动做任何事情，第一次运行训练脚本时会自动处理。

## 3. 训练模型

运行 `train.py` 开始训练模型：

```bash
python3 train.py
```

- 默认配置下，它会训练 100 步（为了快速演示）。
- 训练完成后，模型权重会保存为 `minillm.pt`。
- 如果你想训练一个更好的模型，可以修改 `config.py` 中的 `max_iters`（例如改为 5000）。

## 4. 生成文本

模型训练完成后，运行 `generate.py` 来生成文本：

```bash
python3 generate.py
```

- 脚本会加载 `minillm.pt`。
- 它会使用 "满纸荒唐言，" 作为提示词开始生成。
- 你可以在 `generate.py` 中修改提示词或采样参数（如 `temperature`）。

## 文件说明

- `config.py`: 模型和训练的配置参数。
- `data.py`: 数据加载和预处理逻辑。
- `model.py`: **核心代码**。包含 RMSNorm, RoPE, Attention, SwiGLU 和 Transformer Block 的实现。
- `train.py`: 训练循环。
- `generate.py`: 文本生成脚本。
