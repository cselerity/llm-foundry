"""配置加载器

统一的配置加载接口，支持多种格式。
"""

from pathlib import Path
from typing import Tuple, Union
import yaml

from .model_config import ModelConfig, TrainConfig


def load_config(path: Union[str, Path]) -> Tuple[ModelConfig, TrainConfig]:
    """统一的配置加载器
    
    支持格式:
    - YAML 文件: configs/small.yaml
    - Python 直接创建: ModelConfig(...)
    
    Args:
        path: 配置文件路径
        
    Returns:
        (ModelConfig, TrainConfig) 元组
        
    Example:
        >>> model_cfg, train_cfg = load_config('configs/small.yaml')
        >>> print(model_cfg.dim)
        256
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    
    if path.suffix in ['.yaml', '.yml']:
        return load_yaml_config(path)
    else:
        raise ValueError(f"不支持的配置格式: {path.suffix}")


def load_yaml_config(path: Path) -> Tuple[ModelConfig, TrainConfig]:
    """加载 YAML 配置文件
    
    Args:
        path: YAML 文件路径
        
    Returns:
        (ModelConfig, TrainConfig) 元组
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # 提取模型配置
    model_data = data.get('model', {})
    model_cfg = ModelConfig(**model_data)
    
    # 提取训练配置
    train_data = data.get('training', {})
    train_cfg = TrainConfig(**train_data)
    
    return model_cfg, train_cfg


def save_config(model_cfg: ModelConfig, 
                train_cfg: TrainConfig, 
                path: Union[str, Path]):
    """保存配置到 YAML 文件
    
    Args:
        model_cfg: 模型配置
        train_cfg: 训练配置
        path: 保存路径
        
    Example:
        >>> model_cfg = ModelConfig(dim=256)
        >>> train_cfg = TrainConfig(batch_size=32)
        >>> save_config(model_cfg, train_cfg, 'my_config.yaml')
    """
    path = Path(path)
    
    # 转换为字典
    data = {
        'model': {
            'dim': model_cfg.dim,
            'n_layers': model_cfg.n_layers,
            'n_heads': model_cfg.n_heads,
            'n_kv_heads': model_cfg.n_kv_heads,
            'vocab_size': model_cfg.vocab_size,
            'max_seq_len': model_cfg.max_seq_len,
            'dropout': model_cfg.dropout,
        },
        'training': {
            'batch_size': train_cfg.batch_size,
            'learning_rate': train_cfg.learning_rate,
            'max_iters': train_cfg.max_iters,
            'eval_interval': train_cfg.eval_interval,
            'eval_iters': train_cfg.eval_iters,
        }
    }
    
    # 保存到文件
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    print(f"配置已保存到: {path}")


def get_preset_config(preset: str) -> Tuple[ModelConfig, TrainConfig]:
    """获取预设配置
    
    Args:
        preset: 预设名称 ('small', 'medium', 'rtx5060', 'm4pro')
        
    Returns:
        (ModelConfig, TrainConfig) 元组
        
    Example:
        >>> model_cfg, train_cfg = get_preset_config('small')
    """
    presets = {
        'small': {
            'model': ModelConfig(
                dim=256,
                n_layers=4,
                n_heads=8,
                n_kv_heads=4,
                vocab_size=8192,
                max_seq_len=256,
            ),
            'train': TrainConfig(
                batch_size=32,
                learning_rate=3e-4,
                max_iters=1000,
                eval_interval=50,
                eval_iters=20,
            )
        },
        'medium': {
            'model': ModelConfig(
                dim=512,
                n_layers=8,
                n_heads=8,
                n_kv_heads=4,
                vocab_size=8192,
                max_seq_len=512,
            ),
            'train': TrainConfig(
                batch_size=16,
                learning_rate=3e-4,
                max_iters=5000,
                eval_interval=100,
                eval_iters=50,
            )
        },
        'rtx5060': {
            'model': ModelConfig(
                dim=704,
                n_layers=10,
                n_heads=11,
                n_kv_heads=11,
                vocab_size=8192,
                max_seq_len=512,
            ),
            'train': TrainConfig(
                batch_size=8,
                learning_rate=3e-4,
                max_iters=10000,
                eval_interval=200,
                eval_iters=50,
            )
        },
        'm4pro': {
            'model': ModelConfig(
                dim=704,
                n_layers=10,
                n_heads=11,
                n_kv_heads=11,
                vocab_size=8192,
                max_seq_len=512,
            ),
            'train': TrainConfig(
                batch_size=8,
                learning_rate=3e-4,
                max_iters=10000,
                eval_interval=200,
                eval_iters=50,
            )
        },
    }
    
    if preset not in presets:
        available = ', '.join(presets.keys())
        raise ValueError(f"未知的预设: {preset}. 可用预设: {available}")
    
    config = presets[preset]
    return config['model'], config['train']
