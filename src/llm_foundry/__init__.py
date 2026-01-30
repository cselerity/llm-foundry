"""LLM Foundry: 实用的开源 LLM 基础 —— 从基础到生产

一个轻量级、模块化的 Transformer 语言模型实现,
涵盖从基础概念到生产部署的完整旅程。

主要特性:
- 现代 Transformer 架构 (RoPE, GQA, SwiGLU, RMSNorm)
- 清晰的模块化设计
- 完整的训练和推理流程
- 教育和生产两种使用模式

Modules:
    config: 模型和训练配置
    models: Transformer 模型组件
    tokenizers: 分词器
    data: 数据加载和处理
    training: 训练工具
    inference: 推理和生成
    utils: 实用工具
"""

__version__ = "0.1.0"

from .config import ModelConfig, TrainConfig
from .models import MiniLLM
from .tokenizers import Tokenizer
from .data import DataLoader
from .utils import get_device

__all__ = [
    'ModelConfig',
    'TrainConfig',
    'MiniLLM',
    'Tokenizer',
    'DataLoader',
    'get_device',
]
