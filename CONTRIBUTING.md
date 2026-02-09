# 贡献指南

> **欢迎贡献代码、文档和想法！**

感谢您对 LLM Foundry 项目的关注。本文档将帮助您了解如何参与贡献。

---

## 📋 目录

1. [快速开始](#快速开始)
2. [开发工作流](#开发工作流)
3. [代码规范](#代码规范)
4. [测试指南](#测试指南)
5. [添加新功能](#添加新功能)
6. [文档标准](#文档标准)
7. [提交 Pull Request](#提交-pull-request)

---

## 快速开始

### 环境设置

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

### 分支命名约定

- `main`: 主分支，稳定版本
- `feature/<name>`: 新功能分支 (如 `feature/flash-attention`)
- `fix/<name>`: Bug 修复分支 (如 `fix/rope-overflow`)
- `docs/<name>`: 文档更新分支 (如 `docs/quickstart`)
- `refactor/<name>`: 重构分支 (如 `refactor/data-loader`)

---

## 开发工作流

### 提交信息指南

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

### Pull Request 流程

1. **创建分支**: 从 `main` 创建特性分支
2. **开发**: 实现功能，添加测试，更新文档
3. **提交**: 使用清晰的提交信息
4. **测试**: 运行 `pytest tests/` 确保测试通过
5. **PR**: 创建 Pull Request，描述变更内容
6. **审查**: 等待代码审查，根据反馈修改
7. **合并**: 审查通过后合并到 `main`

### 代码审查清单

- [ ] 代码遵循 PEP 8 风格
- [ ] 有清晰的文档字符串
- [ ] 添加了必要的测试
- [ ] 所有测试通过
- [ ] 更新了相关文档
- [ ] 没有引入性能问题
- [ ] 没有破坏现有功能

---

## 代码规范

### 导入约定

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

### 命名约定

遵循 PEP 8 标准:

- **类名**: `PascalCase` (如 `MiniLLM`, `RMSNorm`)
- **函数/方法**: `snake_case` (如 `get_batch`, `apply_rotary_emb`)
- **常量**: `UPPER_SNAKE_CASE` (如 `MAX_SEQ_LEN`)
- **私有方法**: `_leading_underscore` (如 `_init_weights`)
- **配置类**: `Config` 后缀 (如 `ModelConfig`, `TrainConfig`)

### 文档字符串格式

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

### 代码注释最佳实践

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

---

## 测试指南

### 测试结构

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

### 单元测试要求

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

### 运行测试

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

### 测试覆盖率目标

- 核心模块 (`models`, `data`): **> 80%**
- 工具模块 (`utils`): **> 70%**
- 整体项目: **> 75%**

---

## 添加新功能

### 添加新模型组件

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

5. **更新文档** (`docs/architecture-components.md`):

添加 Flash Attention 的说明和使用示例。

### 添加新训练功能

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

### 添加新数据集

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
4. **更新文档** (`docs/guides/data.md`)

### 添加新采样策略

**场景**: 添加 Top-p 动态调整

**步骤**:

1. **在 `inference/generator.py` 中实现**
2. **添加参数到 `generate` 函数**
3. **在 `examples/03_generation_sampling.py` 中添加示例**
4. **更新文档** (`docs/architecture-training.md`)

---

## 文档标准

### 文档字符串

**必须**: 所有公共 API (类、函数、方法)

**可选**: 私有函数(如果逻辑复杂)

**格式**: Google 风格(见代码规范部分)

### 何时更新文档

**必须更新**:
- 添加新的公共 API
- 修改现有 API 的行为
- 添加新功能或配置选项
- 重大架构变更

**文档文件映射**:
- 模型组件变更 → `docs/architecture-components.md`
- 训练功能变更 → `docs/architecture-training.md`
- 数据处理变更 → `docs/guides/data.md`
- 推理功能变更 → `docs/architecture-training.md`
- 配置变更 → `docs/reference/config.md`
- API 变更 → `docs/reference/api.md`

### README 更新

当以下情况发生时更新 `README.md`:
- 安装方式变更
- 主要功能添加
- 快速入门步骤变化
- 项目目标或定位调整

### 示例创建指南

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

## 提交 Pull Request

### PR 模板

```markdown
## 描述
简要描述此 PR 的目的和变更内容。

## 变更类型
- [ ] Bug 修复
- [ ] 新功能
- [ ] 破坏性变更
- [ ] 文档更新

## 相关 Issue
Closes #(issue number)

## 变更内容
- 列出主要的变更点
- 添加/修改了哪些文件
- 新增了哪些功能

## 测试
- [ ] 添加了单元测试
- [ ] 所有测试通过
- [ ] 手动测试通过

## 文档
- [ ] 更新了相关文档
- [ ] 添加了代码注释
- [ ] 更新了 README (如需要)

## 检查清单
- [ ] 代码遵循项目规范
- [ ] 提交信息清晰
- [ ] 没有引入新的警告
- [ ] 向后兼容 (除非是破坏性变更)
```

### 审查流程

1. **自动检查**: CI 会自动运行测试和代码检查
2. **人工审查**: 维护者会审查代码
3. **反馈修改**: 根据审查意见进行修改
4. **最终合并**: 审查通过后合并

---

## 获取帮助

- 📖 **查看文档**: [docs/README.md](docs/README.md)
- 🐛 **提交 Issue**: [GitHub Issues](https://github.com/your-org/llm-foundry/issues)
- 💬 **讨论**: [GitHub Discussions](https://github.com/your-org/llm-foundry/discussions)
- 📧 **联系**: 在 Discussions 中提问

---

## 项目结构

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
│   ├── architecture/         # 架构详解
│   ├── guides/               # 实用指南
│   ├── hardware/             # 硬件指南
│   └── reference/            # 参考文档
│
├── README.md                 # 项目简介
├── LICENSE                   # MIT 许可证
├── setup.py                  # 包安装
├── requirements.txt          # 依赖
├── requirements-dev.txt      # 开发依赖
└── CONTRIBUTING.md           # 本文档
```

---

## 常见任务快速参考

### 添加新模型组件

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
vim docs/architecture-components.md
```

### 实现新数据集

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
vim docs/guides/data.md
```

### 添加训练功能

```bash
# 1. 在训练模块中实现
vim src/llm_foundry/training/trainer.py

# 2. 更新配置(如果需要)
vim src/llm_foundry/config/model_config.py

# 3. 添加测试
vim tests/test_training.py

# 4. 更新文档
vim docs/architecture-training.md
```

### 创建新示例

```bash
# 1. 创建示例文件
vim examples/06_new_example.py

# 2. 测试运行
python examples/06_new_example.py

# 3. 更新 examples/README.md
vim examples/README.md
```

### 编写文档

```bash
# 1. 创建/编辑文档
vim docs/guides/new_topic.md

# 2. 更新文档索引
vim docs/README.md

# 3. 检查 Markdown 格式
markdownlint new_topic.md

# 4. 预览(可选)
# 使用 Markdown 预览工具
```

### 运行测试

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

**感谢您的贡献！** 🎉
