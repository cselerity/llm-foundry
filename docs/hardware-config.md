# 快速参考卡片

## 🎯 配置速查表

### 预定义配置对比

| 配置 | 参数量 | 显存需求 | 推荐硬件 | 训练时间* |
|------|--------|----------|----------|-----------|
| **Small** | ~2M | <1GB | CPU/笔记本 | 10-30分钟 |
| **Medium** | ~10M | 2-3GB | 入门GPU | 5-10分钟 |
| **Large** | ~70M | 3-4GB | 8GB+ GPU | 2-5分钟 |
| **RTX 5060** | ~70M | 3-4GB | **8GB GPU** | **30-40分钟** |

*训练时间基于 10k steps

---

## 🚀 快速开始

### 1. 小型模型 (教学用)

```python
from config import get_small_config, TrainConfig
from model import MiniLLM

cfg = get_small_config()
model = MiniLLM(cfg)
```

**使用场景**: 快速实验、理解代码、CPU 训练

---

### 2. RTX 5060 优化 (推荐)

```python
from config import get_rtx5060_config, get_rtx5060_train_config

model_cfg = get_rtx5060_config()
train_cfg = get_rtx5060_train_config()
```

**或直接运行:**
```bash
python train_rtx5060.py
```

**使用场景**: 本地学习、实际训练、高质量生成

---

## 📊 模型规格详解

### Small (2M 参数)
```yaml
dim: 256
n_layers: 4
n_heads: 8
n_kv_heads: 4
vocab_size: 8192
max_seq_len: 256
```

### Medium (10M 参数)
```yaml
dim: 512
n_layers: 8
n_heads: 8
n_kv_heads: 4
vocab_size: 16384
max_seq_len: 512
```

### Large / RTX 5060 (70-75M 参数)
```yaml
dim: 768           # BERT-base 同款
n_layers: 12       # 足够深度
n_heads: 12        # 每头 64 维
n_kv_heads: 6      # GQA 优化
vocab_size: 32768  # 32k tokens
max_seq_len: 1024  # 1k 上下文
dropout: 0.1       # 防过拟合
```

---

## ⚙️ 训练参数速查

### 默认训练配置
```yaml
batch_size: 32
learning_rate: 0.001
max_iters: 1000
eval_interval: 100
eval_iters: 20
```

### RTX 5060 优化配置
```yaml
batch_size: 24      # ← 优化显存占用
learning_rate: 3e-4 # ← Adam 推荐
max_iters: 10000    # ← 充分训练
eval_interval: 500
eval_iters: 50
```

---

## 🔧 常用调整

### 显存不足?

```python
# 方案 1: 减小 batch_size
TrainConfig(batch_size=16)

# 方案 2: 减小序列长度
ModelConfig(max_seq_len=512)

# 方案 3: 使用更小的模型
cfg = get_medium_config()
```

### 训练太慢?

```python
# 方案 1: 减少训练步数
TrainConfig(max_iters=5000)

# 方案 2: 增大 batch_size (如果显存够)
TrainConfig(batch_size=32)

# 方案 3: 启用编译 (PyTorch 2.0+)
model = torch.compile(model)
```

### 想要更好的质量?

```python
# 方案 1: 增加训练步数
TrainConfig(max_iters=50000)

# 方案 2: 降低学习率
TrainConfig(learning_rate=1e-4)

# 方案 3: 增大模型
# (需要更多显存)
```

---

## 📈 性能基准

### 不同硬件的 tokens/sec

| 硬件 | Small | Medium | Large/5060 |
|------|-------|--------|------------|
| CPU (8核) | 200-500 | 50-100 | 10-20 |
| RTX 2060 (6GB) | 5k-8k | 2k-3k | 不推荐 |
| RTX 3060 (12GB) | 10k-15k | 4k-6k | 3k-4k |
| **RTX 5060 (8GB)** | **10k-15k** | **4k-6k** | **2.5k-3.5k** |
| RTX 4060 Ti (16GB) | 12k-18k | 5k-7k | 4k-5k |
| RTX 4090 (24GB) | 20k-30k | 10k-15k | 8k-10k |

---

## 💾 显存占用估算

### 公式

```
总显存 = 模型参数 + 优化器 + 激活值

模型参数 = params × 4 bytes  (FP32)
优化器   = params × 8 bytes  (AdamW)
激活值   ≈ batch_size × seq_len × dim × layers × 因子
```

### 实际数据

**Small (2M 参数):**
- 模型: ~8 MB
- 优化器: ~16 MB
- 激活值: ~200 MB
- **总计: ~0.5 GB**

**Medium (10M 参数):**
- 模型: ~40 MB
- 优化器: ~80 MB
- 激活值: ~1 GB
- **总计: ~2 GB**

**Large/RTX5060 (70M 参数):**
- 模型: ~280 MB
- 优化器: ~560 MB
- 激活值: ~2 GB
- **总计: ~3-4 GB**

---

## 🎮 生成参数速查

### Temperature (温度)

```python
temperature = 0.5   # 保守,流畅
temperature = 0.8   # 平衡 (推荐)
temperature = 1.0   # 标准
temperature = 1.5   # 创意,随机
```

### Top-k

```python
top_k = 20    # 非常保守
top_k = 50    # 平衡 (推荐)
top_k = 100   # 更多样化
top_k = None  # 不使用
```

### Top-p (Nucleus)

```python
top_p = 0.9   # 保守
top_p = 0.95  # 平衡 (推荐)
top_p = 0.99  # 更多样化
top_p = None  # 不使用
```

### 常用组合

```python
# 保守生成 (适合正式文本)
generate(..., temperature=0.7, top_k=40, top_p=0.9)

# 平衡生成 (通用推荐)
generate(..., temperature=0.8, top_k=50, top_p=None)

# 创意生成 (适合故事、诗歌)
generate(..., temperature=1.2, top_k=100, top_p=0.95)
```

---

## 🛠️ 常用代码片段

### 1. 完整训练流程

```python
import torch
from config import get_rtx5060_config, get_rtx5060_train_config
from model import MiniLLM
from dataloader import DataLoader

# 配置
model_cfg = get_rtx5060_config()
train_cfg = get_rtx5060_train_config()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据和模型
loader = DataLoader(
    batch_size=train_cfg.batch_size,
    block_size=model_cfg.max_seq_len,
    device=device
)
model = MiniLLM(model_cfg).to(device)

# 训练
optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)

for iter in range(train_cfg.max_iters):
    xb, yb = loader.get_batch('train')
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 保存
torch.save(model.state_dict(), 'model.pt')
```

### 2. 文本生成

```python
import torch
from config import get_rtx5060_config
from model import MiniLLM
from tokenizer import Tokenizer
from generate import generate

# 加载模型
cfg = get_rtx5060_config()
model = MiniLLM(cfg).to('cuda')
model.load_state_dict(torch.load('model.pt'))
model.eval()

# 生成
tokenizer = Tokenizer()
prompt = "红楼梦是"
ids = tokenizer.encode(prompt)
x = torch.tensor(ids).unsqueeze(0).to('cuda')

with torch.no_grad():
    y = generate(model, x, max_new_tokens=100, temperature=0.8, top_k=50)

text = tokenizer.decode(y[0].tolist())
print(text)
```

### 3. 检查点保存和恢复

```python
# 保存完整检查点
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'iter': current_iter,
    'loss': current_loss,
    'config': model_cfg,
}
torch.save(checkpoint, 'checkpoint.pt')

# 恢复训练
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
start_iter = checkpoint['iter']
```

---

## 📝 命令行速查

### GPU 监控

```bash
# 实时监控
watch -n 1 nvidia-smi

# 查看详细显存
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

# 查看进程
nvidia-smi pmon
```

### 训练命令

```bash
# 小型模型 (快速测试)
python train.py

# RTX 5060 优化 (推荐)
python train_rtx5060.py

# 生成文本
python generate.py

# 测试分词器
python tokenizer.py

# 测试数据加载
python dataloader.py
```

### 环境检查

```bash
# 检查 PyTorch 版本
python -c "import torch; print(torch.__version__)"

# 检查 CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 检查 GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"

# 测试 GPU 性能
python -c "import torch; x=torch.rand(1000,1000).cuda(); print(x@x)"
```

---

## 🔗 快速链接

- **完整指南**: [RTX5060_GUIDE.md](RTX5060_GUIDE.md)
- **学习路径**: [../LEARNING_PATH.md](../LEARNING_PATH.md)
- **配置文件**: [config.py](config.py)
- **模型实现**: [model.py](model.py)
- **训练脚本**: [train_rtx5060.py](train_rtx5060.py)

---

## 💡 小贴士

1. **首次训练**: 使用默认 RTX 5060 配置,观察完整流程
2. **调试时**: 使用 Small 配置快速迭代
3. **生产时**: 使用 Large/RTX5060 配置获得最佳质量
4. **显存紧张**: 优先减小 batch_size,其次考虑 max_seq_len
5. **训练时间**: 使用更少的 max_iters 快速验证,满意后再延长

---

**需要详细说明? 查看 [RTX5060_GUIDE.md](RTX5060_GUIDE.md)**
