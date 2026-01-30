"""设备检测工具"""

import torch


def get_device():
    """自动检测可用的计算设备

    Returns:
        str: 'cuda', 'mps', 或 'cpu'
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
