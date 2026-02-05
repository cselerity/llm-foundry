# RTX 5060 训练指南

本指南专门为使用 RTX 5060 (8GB 显存) 或同等性能 GPU 进行本地学习和训练的用户准备。

## 📋 目录

- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [性能优化](#性能优化)
- [故障排除](#故障排除)
- [高级技巧](#高级技巧)

---

## 🚀 快速开始

### 1. 检查环境

```bash
# 检查 CUDA 是否可用
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# 检查显存
nvidia-smi
```

### 2. 运行训练

```bash
cd tutorials
python train_rtx5060.py
```

### 3. 预期结果

- **训练时间**: 30-40 分钟 (10k steps)
- **显存占用**: 3-4 GB
- **训练速度**: 2500-3500 tokens/sec
- **最终损失**: train ~2.0, val ~2.2

---

## ⚙️ 配置说明

### 模型配置 (70-75M 参数)

```python
from config import get_rtx5060_config

model_cfg = get_rtx5060_config()
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `dim` | 768 | 隐藏层维度 (BERT-base 同款) |
| `n_layers` | 12 | Transformer 层数 |
| `n_heads` | 12 | 注意力头数 (每头 64 维) |
| `n_kv_heads` | 6 | KV 头数 (GQA 优化,节省 50% 显存) |
| `vocab_size` | 32768 | 词汇表大小 (32k tokens) |
| `max_seq_len` | 1024 | 最大序列长度 |
| `dropout` | 0.1 | Dropout 率 |

### 训练配置

```python
from config import get_rtx5060_train_config

train_cfg = get_rtx5060_train_config()
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `batch_size` | 24 | 批次大小 (优化的显存占用) |
| `learning_rate` | 3e-4 | 学习率 (Adam 推荐值) |
| `max_iters` | 10000 | 训练步数 |
| `eval_interval` | 500 | 评估间隔 |
| `eval_iters` | 50 | 评估批次数 |

---

## 📊 性能预期

### 不同 GPU 的性能对比

| GPU 型号 | 显存 | Batch Size | 训练速度 | 10k Steps 时间 |
|----------|------|------------|----------|----------------|
| RTX 3060 | 12GB | 32-48 | 3000-4000 tok/s | 25-35 分钟 |
| **RTX 5060** | **8GB** | **24-32** | **2500-3500 tok/s** | **30-40 分钟** |
| RTX 4060 | 8GB | 24-32 | 2500-3500 tok/s | 30-40 分钟 |
| RTX 4060 Ti | 16GB | 48-64 | 4000-5000 tok/s | 20-30 分钟 |
| RTX 4090 | 24GB | 96-128 | 8000-10000 tok/s | 10-15 分钟 |

### 显存占用详细分解

```
总显存占用: 3-4 GB (训练时)
├── 模型参数:   ~0.28 GB (70M × 4 bytes)
├── 优化器状态: ~0.56 GB (AdamW, 2 状态)
├── 激活值:     ~1.5-2 GB (取决于 batch_size)
└── KV cache:   ~0.3 GB (GQA 优化后)
```

---

## 🔧 性能优化

### 1. 调整 Batch Size

**如果显存不足:**

```python
# 在 config.py 中修改
TrainConfig(
    batch_size=16,  # 减小到 16
    # ... 其他参数
)
```

**如果显存充足:**

```python
# 尝试更大的 batch size
TrainConfig(
    batch_size=32,  # 增加到 32
    # ... 其他参数
)
```

### 2. 调整序列长度

**减少序列长度可以显著节省显存:**

```python
ModelConfig(
    max_seq_len=512,  # 从 1024 减到 512
    # ... 其他参数
)
```

### 3. 使用梯度累积

**如果想要更大的有效 batch size 但显存不足:**

```python
# 在训练循环中添加
accumulation_steps = 2  # 累积 2 步再更新

for iter in range(max_iters):
    for micro_step in range(accumulation_steps):
        xb, yb = loader.get_batch('train')
        logits, loss = model(xb, yb)
        loss = loss / accumulation_steps  # 缩放损失
        loss.backward()

    optimizer.step()
    optimizer.zero_grad()
```

### 4. 使用混合精度训练

**可以节省约 40% 显存:**

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for iter in range(max_iters):
    xb, yb = loader.get_batch('train')

    with autocast():  # 自动混合精度
        logits, loss = model(xb, yb)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### 5. 启用 PyTorch 编译 (PyTorch 2.0+)

**可以提升 20-30% 速度:**

```python
import torch

model = MiniLLM(cfg).to('cuda')
model = torch.compile(model)  # 编译模型
```

---

## 🐛 故障排除

### 问题 1: CUDA out of memory

**症状:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**解决方案:**

1. **减小 batch_size:**
   ```python
   TrainConfig(batch_size=16)  # 或更小
   ```

2. **减小序列长度:**
   ```python
   ModelConfig(max_seq_len=512)
   ```

3. **清理 GPU 缓存:**
   ```python
   torch.cuda.empty_cache()
   ```

4. **关闭其他占用 GPU 的程序:**
   ```bash
   nvidia-smi  # 查看 GPU 占用
   kill <pid>  # 关闭占用进程
   ```

### 问题 2: 训练速度很慢

**可能原因和解决方案:**

1. **没有使用 GPU:**
   ```bash
   # 检查
   python -c "import torch; print(torch.cuda.is_available())"

   # 如果返回 False,重新安装 CUDA 版本的 PyTorch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

2. **GPU 被其他程序占用:**
   ```bash
   nvidia-smi  # 检查 GPU 利用率应该接近 100%
   ```

3. **数据加载成为瓶颈:**
   ```python
   # 增加数据加载工作进程 (如果使用 DataLoader)
   DataLoader(..., num_workers=4)
   ```

4. **PyTorch 版本过旧:**
   ```bash
   pip install --upgrade torch
   ```

### 问题 3: 损失为 NaN

**可能原因和解决方案:**

1. **学习率过高:**
   ```python
   TrainConfig(learning_rate=1e-4)  # 降低学习率
   ```

2. **梯度爆炸:**
   ```python
   # 添加梯度裁剪
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

3. **数据问题:**
   ```python
   # 检查数据是否有异常值
   print(f"Data range: {loader.tokens.min()} - {loader.tokens.max()}")
   ```

### 问题 4: 过拟合

**症状:**
```
train loss: 1.5, val loss: 2.8  (差距过大)
```

**解决方案:**

1. **增加 Dropout:**
   ```python
   ModelConfig(dropout=0.2)  # 从 0.1 增加到 0.2
   ```

2. **减少训练步数:**
   ```python
   TrainConfig(max_iters=5000)  # 减少迭代次数
   ```

3. **使用更多训练数据:**
   - 添加更多文本到训练集
   - 使用数据增强

4. **减小模型规模:**
   ```python
   # 使用 medium 配置
   from config import get_medium_config
   model_cfg = get_medium_config()
   ```

---

## 🎯 高级技巧

### 1. 学习率预热 (Learning Rate Warmup)

**逐渐增加学习率,训练更稳定:**

```python
def get_lr(iter, warmup_iters=1000, lr_decay_iters=10000, min_lr=1e-5):
    # Warmup
    if iter < warmup_iters:
        return learning_rate * iter / warmup_iters
    # Decay
    if iter > lr_decay_iters:
        return min_lr
    # Cosine decay
    decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# 在训练循环中使用
for iter in range(max_iters):
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # ... 训练步骤
```

### 2. 检查点保存和恢复

**定期保存检查点,避免训练中断:**

```python
# 保存检查点
if iter % 1000 == 0:
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter': iter,
        'config': model_cfg,
    }
    torch.save(checkpoint, f'checkpoint_{iter}.pt')

# 恢复训练
checkpoint = torch.load('checkpoint_5000.pt')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
start_iter = checkpoint['iter']
```

### 3. 早停 (Early Stopping)

**验证损失不再下降时自动停止:**

```python
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(...)

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping!")
                break
```

### 4. 使用 TensorBoard 监控

**可视化训练过程:**

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/rtx5060_experiment')

for iter in range(max_iters):
    # ... 训练步骤

    if iter % eval_interval == 0:
        losses = estimate_loss(...)
        writer.add_scalar('Loss/train', losses['train'], iter)
        writer.add_scalar('Loss/val', losses['val'], iter)
        writer.add_scalar('LearningRate', lr, iter)

writer.close()
```

然后启动 TensorBoard:
```bash
tensorboard --logdir=runs
```

### 5. 批量生成和评估

**评估模型生成质量:**

```python
def generate_samples(model, tokenizer, prompts, max_tokens=100):
    """批量生成文本样本"""
    model.eval()
    samples = []

    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        x = torch.tensor(ids).unsqueeze(0).to(device)

        with torch.no_grad():
            y = generate(model, x, max_tokens, temperature=0.8, top_k=50)

        text = tokenizer.decode(y[0].tolist())
        samples.append(text)

    return samples

# 使用
prompts = ["红楼梦", "人工智能", "从前有座山"]
samples = generate_samples(model, tokenizer, prompts)
for prompt, sample in zip(prompts, samples):
    print(f"Prompt: {prompt}")
    print(f"Generated: {sample}\n")
```

---

## 📈 性能基准测试

### 运行基准测试

```bash
# 测试不同 batch size 的性能
for bs in 16 24 32; do
    echo "Testing batch_size=$bs"
    python -c "
from config import ModelConfig, TrainConfig
config = TrainConfig(batch_size=$bs)
# ... 运行训练
    "
done
```

### 预期结果

| Batch Size | 显存占用 | 速度 (tokens/s) | 备注 |
|------------|----------|-----------------|------|
| 16 | 2.5-3 GB | 2200-2800 | 安全选择 |
| **24** | **3-4 GB** | **2500-3500** | **推荐 (平衡)** |
| 32 | 4-5 GB | 2800-4000 | 需要足够显存 |

---

## 🎓 学习建议

### 循序渐进的学习路径

1. **第一次训练** (10-30 分钟):
   - 使用默认 RTX 5060 配置
   - 观察训练过程和日志输出
   - 理解各项指标的含义

2. **实验参数** (1-2 小时):
   - 调整 batch_size: 16, 24, 32
   - 调整 learning_rate: 1e-4, 3e-4, 5e-4
   - 观察对训练速度和损失的影响

3. **优化显存** (30 分钟):
   - 尝试减小 max_seq_len: 512, 768, 1024
   - 测试混合精度训练
   - 使用梯度累积

4. **提升质量** (2-3 小时):
   - 增加训练步数: 20k, 50k
   - 使用更大的数据集
   - 实验不同的采样策略

5. **生产优化** (进阶):
   - 实现学习率调度
   - 添加早停机制
   - 使用 TensorBoard 监控

---

## 📚 参考资料

### 相关文件

- [config.py](config.py) - 配置定义
- [train_rtx5060.py](train_rtx5060.py) - RTX 5060 训练脚本
- [model.py](model.py) - 模型实现
- [LEARNING_PATH.md](../LEARNING_PATH.md) - 学习路径

### 有用的命令

```bash
# 监控 GPU 使用
watch -n 1 nvidia-smi

# 查看 PyTorch 版本
python -c "import torch; print(torch.__version__)"

# 测试 CUDA 性能
python -c "import torch; x = torch.rand(1000, 1000).cuda(); print(x @ x)"

# 查看显存占用详情
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv
```

---

## ❓ 常见问题

### Q1: 我的 GPU 不是 RTX 5060,也能用这个配置吗?

**A:** 可以! 这个配置适用于大多数 8GB 显存的 GPU,包括:
- RTX 3060 (12GB) - 可以增大 batch_size
- RTX 4060 (8GB) - 完全兼容
- RTX 2060 Super (8GB) - 可能需要稍微调整
- GTX 1080 Ti (11GB) - 可以增大 batch_size

### Q2: 训练需要多久才能看到好的结果?

**A:**
- **初步结果**: 1000-2000 steps (3-5 分钟)
- **可用质量**: 5000-10000 steps (15-30 分钟)
- **良好质量**: 20000-50000 steps (1-2 小时)

### Q3: 如何知道训练是否正常?

**A:** 观察以下指标:
- **训练损失下降**: 应该从 ~10 降到 ~2
- **验证损失跟随**: 不应该远高于训练损失
- **GPU 利用率**: nvidia-smi 应该显示 90-100%
- **生成质量**: 定期测试生成文本

### Q4: 可以在训练时使用电脑吗?

**A:** 可以,但建议:
- 不要同时运行其他 GPU 密集任务
- 浏览器和轻量应用没问题
- 可以在后台运行训练,使用 screen 或 tmux

### Q5: 如何提高生成质量?

**A:** 尝试:
1. 增加训练步数 (max_iters)
2. 使用更大的模型 (但需要更多显存)
3. 增加训练数据
4. 调整采样参数 (temperature, top_k, top_p)
5. 使用更长的提示词 (prompt)

---

## 💡 贡献

发现问题或有改进建议? 欢迎:
- 提交 Issue
- 创建 Pull Request
- 分享你的训练经验

---

**祝训练愉快! 🚀**
