# AGENTS.md - Agent 协作指南

> **LLM Foundry**: 实用的开源 LLM 基础 —— 从基础到生产

本文档为 AI Agent 提供全面的项目导航和协作指南,帮助理解项目结构、代码组织和开发工作流。

---

## 1. 项目概览

### 1.1 使命与愿景

**使命**: 提供一个实用的、开源的 LLM 基础实现,覆盖从基础概念到生产部署的完整旅程。

**愿景**:
- 🎓 **教育性**: 清晰的代码结构和详细的注释,适合学习
- 🏭 **生产性**: 模块化设计,易于扩展和部署
- 🌉 **桥梁性**: 连接理论学习和实际应用

### 1.2 目标受众

- ML 工程师:需要理解和定制 LLM 实现
- 研究人员:探索 Transformer 架构和训练技术
- 学生:学习现代 LLM 的工作原理

### 1.3 核心原则

1. **简洁性**: 代码简洁易懂,避免过度工程化
2. **模块化**: 清晰的模块边界,便于测试和复用
3. **可扩展性**: 易于添加新功能和改进
4. **文档化**: 完整的文档和注释

---

## 2. 架构概览

### 2.1 目录结构

```
llm-foundry/
├── src/llm_foundry/          # 主包 - 生产级代码
│   ├── config/               # 配置模块
│   ├── models/               # 模型实现
│   ├── tokenizers/           # 分词器
│   ├── data/                 # 数据处理
│   ├── training/             # 训练工具
│   ├── inference/            # 推理工具
│   └── utils/                # 实用工具
│
├── tutorials/                # 教学展示 - 核心功能的完整展示
│   ├── train.py              # 教学训练脚本
│   ├── generate.py           # 教学生成脚本
│   └── README.md             # 简单模式说明
│
├── scripts/                  # 命令行工具
│   ├── train.py              # 训练入口
│   ├── generate.py           # 生成入口
│   ├── evaluate.py           # 评估工具
│   └── prepare_data.py       # 数据准备
│
├── examples/                 # 使用示例
│   ├── 01_basic_training.py
│   ├── 02_custom_data.py
│   ├── 03_generation_sampling.py
│   ├── 04_fine_tuning.py
│   └── datasets/             # 数据集下载器
│
├── tests/                    # 单元测试
├── configs/                  # 配置文件
├── docs/                     # 文档
│   ├── zh/                   # 中文文档
│   │   ├── quickstart.md
│   │   ├── architecture.md
│   │   ├── training.md
│   │   ├── inference.md
│   │   └── production/       # 生产部署指南
│   └── assets/               # 图片资源
│
├── README.md                 # 项目简介
├── LICENSE                   # MIT 许可证
├── setup.py                  # 包安装
├── requirements.txt          # 依赖
└── AGENTS.md                 # 本文档
```

### 2.2 模块职责

| 模块 | 职责 | 关键文件 |
|------|------|---------|
| `config` | 配置管理 | `model_config.py` - 模型和训练配置 |
| `models` | 模型实现 | `components.py` - 基础组件<br>`transformer.py` - 完整模型 |
| `tokenizers` | 文本分词 | `sp_tokenizer.py` - SentencePiece BPE |
| `data` | 数据处理 | `loader.py` - 数据加载和批次生成 |
| `training` | 训练流程 | `trainer.py` - 训练器类 |
| `inference` | 文本生成 | `generator.py` - 生成器类 |
| `utils` | 工具函数 | `device.py` - 设备检测<br>`checkpointing.py` - 检查点管理 |

### 2.3 双模式设计

**教学展示** (`tutorials/`):
- 主工程核心功能的完整展示
- 单文件脚本,易于理解
- 适合快速实验和教学
- 功能与工程版本对等

**工程实现** (`src/llm_foundry/`):
- 模块化架构,生产就绪
- 支持 `pip install`
- 便于扩展和维护

---

## 3. 代码组织

### 3.1 包结构

```python
# src/llm_foundry/ 的导入层次

llm_foundry/
├── __init__.py           # 顶层导出
├── config/
│   ├── __init__.py       # 导出: ModelConfig, TrainConfig
│   └── model_config.py   # 配置类定义
├── models/
│   ├── __init__.py       # 导出: MiniLLM, RMSNorm, etc.
│   ├── components.py     # 基础组件
│   └── transformer.py    # 完整模型
└── ...
```

### 3.2 导入约定

**推荐的导入方式**:

```python
# 1. 顶层导入(简单使用)
from llm_foundry import ModelConfig, MiniLLM, Tokenizer

# 2. 模块导入(明确来源)
from llm_foundry.models import MiniLLM
from llm_foundry.config import ModelConfig, TrainConfig

# 3. 内部导入(包内使用)
from ..config import ModelConfig
from .components import RMSNorm
```

**避免的导入方式**:

```python
# ❌ 避免 import *
from llm_foundry import *

# ❌ 避免深层导入(破坏封装)
from llm_foundry.models.components import RMSNorm  # 应该从 models 导入
```

### 3.3 命名约定

遵循 PEP 8 标准:

- **类名**: `PascalCase` (如 `MiniLLM`, `RMSNorm`)
- **函数/方法**: `snake_case` (如 `get_batch`, `apply_rotary_emb`)
- **常量**: `UPPER_SNAKE_CASE` (如 `MAX_SEQ_LEN`)
- **私有方法**: `_leading_underscore` (如 `_init_weights`)
- **配置类**: `Config` 后缀 (如 `ModelConfig`, `TrainConfig`)

### 3.4 文档字符串格式

使用 Google 风格的文档字符串:

```python
def function_name(arg1, arg2):
    """简短描述(一行)

    详细描述(可选,多行)。
    解释函数的行为、算法或设计决策。

    Args:
        arg1: 参数 1 的描述
        arg2: 参数 2 的描述

    Returns:
        返回值的描述

    Raises:
        ExceptionType: 异常情况的描述

    Example:
        >>> result = function_name(1, 2)
        >>> print(result)
        3
    """
    pass
```

---

## 4. 开发工作流

### 4.1 环境设置

```bash
# 1. 克隆仓库
git clone https://github.com/your-org/llm-foundry.git
cd llm-foundry

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -e .  # 开发模式安装
pip install -r requirements-dev.txt  # 开发工具

# 4. 验证安装
python -c "import llm_foundry; print(llm_foundry.__version__)"
```

### 4.2 分支命名约定

- `main`: 主分支,稳定版本
- `feature/<name>`: 新功能分支 (如 `feature/flash-attention`)
- `fix/<name>`: Bug 修复分支 (如 `fix/rope-overflow`)
- `docs/<name>`: 文档更新分支 (如 `docs/quickstart`)
- `refactor/<name>`: 重构分支 (如 `refactor/data-loader`)

### 4.3 提交信息指南

格式: `<type>(<scope>): <subject>`

**类型**:
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建/工具配置

**示例**:
```
feat(models): add Flash Attention support
fix(data): handle empty tokenizer files
docs(zh): update quickstart guide
refactor(training): extract loss computation
```

### 4.4 Pull Request 流程

1. **创建分支**: 从 `main` 创建特性分支
2. **开发**: 实现功能,添加测试,更新文档
3. **提交**: 使用清晰的提交信息
4. **测试**: 运行 `pytest tests/` 确保测试通过
5. **PR**: 创建 Pull Request,描述变更内容
6. **审查**: 等待代码审查,根据反馈修改
7. **合并**: 审查通过后合并到 `main`

### 4.5 代码审查清单

- [ ] 代码遵循 PEP 8 风格
- [ ] 有清晰的文档字符串
- [ ] 添加了必要的测试
- [ ] 所有测试通过
- [ ] 更新了相关文档
- [ ] 没有引入性能问题
- [ ] 没有破坏现有功能

---

## 5. 添加新功能

### 5.1 添加新模型组件

**场景**: 添加新的注意力机制(如 Flash Attention)

**步骤**:

1. **在 `models/components.py` 中定义**:

```python
class FlashAttention(nn.Module):
    """Flash Attention 实现

    快速且内存高效的注意力实现。
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # 实现细节...

    def forward(self, x, freqs_cis):
        # 前向传播逻辑...
        pass
```

2. **在 `models/__init__.py` 中导出**:

```python
from .components import FlashAttention

__all__ = [
    # ... 现有导出
    'FlashAttention',
]
```

3. **在 `models/transformer.py` 中使用**:

```python
# 可选地在 Block 中使用新组件
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, use_flash=False):
        super().__init__()
        if use_flash:
            self.attention = FlashAttention(cfg)
        else:
            self.attention = CausalSelfAttention(cfg)
```

4. **添加测试** (`tests/test_models.py`):

```python
def test_flash_attention():
    cfg = ModelConfig()
    attn = FlashAttention(cfg)
    x = torch.randn(2, 16, cfg.dim)
    freqs_cis = precompute_freqs_cis(cfg.dim // cfg.n_heads, 16)
    output = attn(x, freqs_cis)
    assert output.shape == x.shape
```

5. **更新文档** (`docs/zh/architecture.md`):

添加 Flash Attention 的说明和使用示例。

### 5.2 添加新训练功能

**场景**: 添加学习率调度器

**步骤**:

1. **在 `training/` 中创建新文件** (`schedulers.py`):

```python
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """余弦退火学习率调度器"""
    # 实现...
    pass
```

2. **在 `training/__init__.py` 中导出**
3. **在 `Trainer` 类中集成**
4. **添加配置选项** (`TrainConfig`)
5. **更新文档和示例**

### 5.3 添加新数据集

**场景**: 添加英文文本数据集支持

**步骤**:

1. **在 `data/datasets.py` 中定义**:

```python
class TinyStoriesDataset:
    """TinyStories 数据集加载器"""
    def __init__(self, cache_dir='./data'):
        # 实现...
        pass
```

2. **在 `examples/datasets/` 中添加下载器**:

```python
# download_english.py
def download_tinystories():
    """下载 TinyStories 数据集"""
    # 实现...
```

3. **添加使用示例** (`examples/02_custom_data.py`)
4. **更新文档** (`docs/zh/data-preparation.md`)

### 5.4 添加新采样策略

**场景**: 添加 Top-p 动态调整

**步骤**:

1. **在 `inference/generator.py` 中实现**
2. **添加参数到 `generate` 函数**
3. **在 `examples/03_generation_sampling.py` 中添加示例**
4. **更新文档** (`docs/zh/inference.md`)

---

## 6. 测试指南

### 6.1 测试结构

```
tests/
├── __init__.py
├── test_models.py        # 模型组件测试
├── test_tokenizer.py     # 分词器测试
├── test_data.py          # 数据加载测试
├── test_training.py      # 训练流程测试
└── fixtures/             # 测试数据
    └── sample.txt
```

### 6.2 单元测试要求

**每个新功能都应该有测试**:

```python
import pytest
import torch
from llm_foundry import ModelConfig
from llm_foundry.models import RMSNorm

def test_rmsnorm_shape():
    """测试 RMSNorm 输出形状"""
    cfg = ModelConfig()
    norm = RMSNorm(cfg.dim)
    x = torch.randn(2, 16, cfg.dim)
    output = norm(x)
    assert output.shape == x.shape

def test_rmsnorm_normalization():
    """测试 RMSNorm 归一化效果"""
    cfg = ModelConfig()
    norm = RMSNorm(cfg.dim)
    x = torch.randn(2, 16, cfg.dim)
    output = norm(x)
    # 验证归一化属性
    rms = torch.sqrt(torch.mean(output ** 2, dim=-1))
    # RMS 应该接近 1
    assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)
```

### 6.3 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定文件
pytest tests/test_models.py

# 运行特定测试
pytest tests/test_models.py::test_rmsnorm_shape

# 显示详细输出
pytest tests/ -v

# 显示打印输出
pytest tests/ -s

# 生成覆盖率报告
pytest tests/ --cov=llm_foundry --cov-report=html
```

### 6.4 测试覆盖率目标

- 核心模块 (`models`, `data`): **> 80%**
- 工具模块 (`utils`): **> 70%**
- 整体项目: **> 75%**

---

## 7. 文档标准

### 7.1 文档字符串

**必须**: 所有公共 API (类、函数、方法)

**可选**: 私有函数(如果逻辑复杂)

**格式**: Google 风格(见 3.4 节)

### 7.2 何时更新 docs/zh/

**必须更新**:
- 添加新的公共 API
- 修改现有 API 的行为
- 添加新功能或配置选项
- 重大架构变更

**文档文件映射**:
- 模型组件变更 → `architecture.md`
- 训练功能变更 → `training.md`
- 数据处理变更 → `data-preparation.md`
- 推理功能变更 → `inference.md`
- 配置变更 → `configuration.md`
- API 变更 → `api-reference.md`

### 7.3 README 更新

当以下情况发生时更新 `README.md`:
- 安装方式变更
- 主要功能添加
- 快速入门步骤变化
- 项目目标或定位调整

### 7.4 代码注释最佳实践

**好的注释**:
```python
# 使用 RoPE 而不是绝对位置编码,因为它对长序列外推效果更好
xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

# 对于 GQA,重复 KV heads 以匹配 query heads 的数量
# 例如: 8 query heads, 4 KV heads -> 每个 KV head 重复 2 次
if self.n_kv_heads != self.n_heads:
    xk = torch.repeat_interleave(xk, self.n_heads // self.n_kv_heads, dim=2)
```

**不好的注释**:
```python
# 应用 RoPE
xq, xk = apply_rotary_emb(xq, xk, freqs_cis)  # ❌ 重复代码含义

# i = i + 1
i = i + 1  # ❌ 无意义注释
```

### 7.5 示例创建指南

在 `examples/` 中创建新示例时:

1. **命名**: 使用数字前缀排序 (如 `05_new_feature.py`)
2. **结构**:
   - 简短的文档字符串说明目的
   - 清晰的步骤注释
   - 完整的可运行代码
   - 输出示例(在注释中)

3. **模板**:

```python
"""05_new_feature.py - 新功能示例

演示如何使用 XXX 功能来实现 YYY。

运行:
    python examples/05_new_feature.py
"""

import torch
from llm_foundry import ModelConfig, MiniLLM

def main():
    # 1. 设置配置
    cfg = ModelConfig()
    print(f"使用配置: {cfg}")

    # 2. 创建模型
    model = MiniLLM(cfg)

    # 3. 演示功能
    # ...

    print("示例完成!")

if __name__ == '__main__':
    main()
```

---

## 8. Agent 协作模式

### 8.1 如何高效导航代码库

**查找功能的位置**:

| 需求 | 位置 | 方法 |
|------|------|------|
| 模型架构细节 | `src/llm_foundry/models/` | 查看 `components.py` 和 `transformer.py` |
| 训练逻辑 | `src/llm_foundry/training/` | 查看 `trainer.py` |
| 数据处理流程 | `src/llm_foundry/data/` | 查看 `loader.py` |
| 配置选项 | `src/llm_foundry/config/` | 查看 `model_config.py` |
| 生成逻辑 | `src/llm_foundry/inference/` | 查看 `generator.py` |
| 使用示例 | `examples/` | 浏览编号文件 |
| 教学脚本 | `tutorials/` | 查看 `train.py` 和 `generate.py` |

**使用 Grep 搜索**:

```bash
# 查找函数定义
grep -r "def generate" src/

# 查找类定义
grep -r "class MiniLLM" src/

# 查找配置使用
grep -r "ModelConfig" src/

# 查找 TODO 注释
grep -r "TODO" src/
```

### 8.2 常见修改模式

**模式 1: 修改超参数**
1. 更新 `config/model_config.py` 中的默认值
2. 或在 `configs/*.yaml` 中添加新配置文件
3. 更新文档 `docs/zh/configuration.md`

**模式 2: 优化现有组件**
1. 修改 `models/components.py` 中的实现
2. 保持接口不变(向后兼容)
3. 添加/更新测试
4. 更新文档(如果行为改变)

**模式 3: 添加新功能**
1. 在适当模块中实现
2. 更新模块的 `__init__.py`
3. 添加测试
4. 创建使用示例
5. 更新文档

### 8.3 模块间依赖关系

```
依赖方向: 高层 → 低层

应用层:      scripts/, tutorials/, examples/
                ↓
高层 API:     training/, inference/
                ↓
核心模块:     models/, data/, tokenizers/
                ↓
基础模块:     config/, utils/
```

**原则**:
- 低层模块不应依赖高层模块
- 避免循环依赖
- 使用接口(配置类)解耦

### 8.4 安全重构实践

**重构检查清单**:

1. **理解现有代码**:
   - 阅读相关模块的代码
   - 理解功能和边界条件
   - 查看现有测试

2. **制定计划**:
   - 明确重构目标
   - 识别受影响的模块
   - 计划向后兼容策略

3. **增量重构**:
   - 小步骤提交
   - 每步后运行测试
   - 保持功能不变

4. **更新相关内容**:
   - 更新导入语句
   - 更新文档
   - 更新示例

5. **验证**:
   - 运行完整测试套件
   - 运行示例脚本
   - 检查文档准确性

---

## 9. 生产环境考虑

### 9.1 性能优化指南

**关键优化点**:

1. **模型效率**:
   - 使用 `F.scaled_dot_product_attention`(PyTorch 2.0+)
   - 启用 `torch.compile`(可选)
   - 使用混合精度训练

2. **数据加载**:
   - 使用 `DataLoader` 的 `num_workers`
   - 预加载数据到内存(如果可能)
   - 使用 `pin_memory=True` (CUDA)

3. **内存管理**:
   - 使用梯度累积处理大批次
   - 及时释放不需要的张量
   - 使用 `torch.cuda.empty_cache()` 清理缓存

**性能分析**:

```python
# 使用 PyTorch Profiler
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # 运行训练/推理代码
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### 9.2 内存效率模式

**梯度累积**:

```python
# 有效批次大小 = batch_size * accumulation_steps
accumulation_steps = 4

for i, (x, y) in enumerate(dataloader):
    logits, loss = model(x, y)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**梯度检查点**:

```python
from torch.utils.checkpoint import checkpoint

# 在 Block 的 forward 中
def forward(self, x, freqs_cis):
    # 使用检查点节省内存(牺牲一些速度)
    h = x + checkpoint(self.attention, self.attention_norm(x), freqs_cis)
    out = h + checkpoint(self.feed_forward, self.ffn_norm(h))
    return out
```

### 9.3 分布式训练

**DDP 示例**:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train_ddp(rank, world_size):
    setup(rank, world_size)
    model = MiniLLM(cfg).to(rank)
    model = DDP(model, device_ids=[rank])
    # 训练代码...
```

**使用指南**: 查看 `docs/zh/production/distributed-training.md`

### 9.4 模型服务

**FastAPI 服务示例**:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = None  # 全局模型

@app.on_event("startup")
async def load_model():
    global model
    model = MiniLLM(cfg)
    model.load_state_dict(torch.load("minillm.pt"))
    model.eval()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100

@app.post("/generate")
async def generate(request: GenerateRequest):
    # 生成逻辑...
    return {"text": generated_text}
```

**使用指南**: 查看 `docs/zh/production/model-serving.md`

### 9.5 监控和日志

**训练监控**:

```python
# 使用 tensorboard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')

for epoch in range(num_epochs):
    # 训练...
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Learning_rate', lr, epoch)

writer.close()
```

**日志记录**:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Training started")
```

---

## 10. 常见任务快速参考

### 10.1 添加新模型组件

```bash
# 1. 编辑组件文件
vim src/llm_foundry/models/components.py

# 2. 更新 __init__.py
vim src/llm_foundry/models/__init__.py

# 3. 添加测试
vim tests/test_models.py

# 4. 运行测试
pytest tests/test_models.py -v

# 5. 更新文档
vim docs/zh/architecture.md
```

### 10.2 实现新数据集

```bash
# 1. 创建数据集类
vim src/llm_foundry/data/datasets.py

# 2. 创建下载器
vim examples/datasets/download_new.py

# 3. 添加测试
vim tests/test_data.py

# 4. 创建示例
vim examples/05_new_dataset.py

# 5. 更新文档
vim docs/zh/data-preparation.md
```

### 10.3 添加训练功能

```bash
# 1. 在训练模块中实现
vim src/llm_foundry/training/trainer.py

# 2. 更新配置(如果需要)
vim src/llm_foundry/config/model_config.py

# 3. 添加测试
vim tests/test_training.py

# 4. 更新文档
vim docs/zh/training.md
```

### 10.4 创建新示例

```bash
# 1. 创建示例文件
vim examples/06_new_example.py

# 2. 测试运行
python examples/06_new_example.py

# 3. 更新 examples/README.md
vim examples/README.md
```

### 10.5 编写文档

```bash
# 1. 创建/编辑文档
vim docs/zh/new_topic.md

# 2. 更新文档索引
vim docs/README.md

# 3. 检查 Markdown 格式
markdownlint docs/zh/new_topic.md

# 4. 预览(可选)
# 使用 Markdown 预览工具
```

### 10.6 运行测试

```bash
# 快速测试(核心功能)
pytest tests/test_models.py tests/test_data.py

# 完整测试
pytest tests/ -v --cov=llm_foundry

# 测试特定功能
pytest tests/ -k "test_rmsnorm"

# 生成覆盖率报告
pytest tests/ --cov=llm_foundry --cov-report=html
open htmlcov/index.html
```

---

## 总结

本文档为 AI Agent 提供了全面的项目导航指南。关键要点:

1. **项目使命**: 从基础到生产的实用 LLM 基础
2. **双模式设计**: 简单模式(教学)+ 包模式(生产)
3. **清晰的模块划分**: config, models, data, training, inference, utils
4. **完整的开发工作流**: 从环境设置到 PR 合并
5. **生产就绪**: 性能优化、分布式训练、模型服务

**下一步**:
- 阅读 `docs/zh/quickstart.md` 快速上手
- 查看 `examples/` 了解使用方式
- 浏览 `docs/zh/architecture.md` 深入理解架构

**保持联系**:
- 问题反馈: GitHub Issues
- 讨论交流: GitHub Discussions
- 贡献代码: Pull Requests

---

**版本**: 0.1.0
**最后更新**: 2026-01-30
**许可证**: MIT
