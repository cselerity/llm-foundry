"""数据处理模块

提供数据加载、预处理和批次生成功能。
"""

from .loader import DataLoader, download_data

__all__ = ['DataLoader', 'download_data']
