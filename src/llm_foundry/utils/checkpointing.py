"""模型检查点保存和加载工具"""

import torch
import os


def save_checkpoint(model, filepath, optimizer=None, **kwargs):
    """保存模型检查点

    Args:
        model: PyTorch 模型
        filepath: 保存路径
        optimizer: 优化器(可选)
        **kwargs: 其他要保存的信息
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        **kwargs
    }
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"检查点已保存到 {filepath}")


def load_checkpoint(model, filepath, optimizer=None, device='cpu'):
    """加载模型检查点

    Args:
        model: PyTorch 模型
        filepath: 检查点文件路径
        optimizer: 优化器(可选)
        device: 加载到的设备

    Returns:
        dict: 检查点中的其他信息
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 返回其他保存的信息
    other_info = {k: v for k, v in checkpoint.items()
                  if k not in ['model_state_dict', 'optimizer_state_dict']}

    print(f"检查点已从 {filepath} 加载")
    return other_info
