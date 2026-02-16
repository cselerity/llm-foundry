"""配置模块

包含模型和训练的配置类。
"""

from .model_config import ModelConfig, TrainConfig
from .loader import load_config, save_config, get_preset_config

__all__ = [
    'ModelConfig', 
    'TrainConfig',
    'load_config',
    'save_config',
    'get_preset_config',
]
