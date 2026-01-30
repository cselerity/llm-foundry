"""数据加载器

提供数据下载、分词和批次生成功能。
"""

import os
import requests
import torch
import numpy as np

from ..config import TrainConfig
from ..tokenizers import Tokenizer


def download_data(file_path='input_cn.txt', url=None):
    """下载数据集

    如果文件不存在,则从指定 URL 下载数据。
    默认下载红楼梦数据集。

    Args:
        file_path: 保存文件路径
        url: 下载 URL,默认为红楼梦数据集
    """
    if not os.path.exists(file_path):
        if url is None:
            url = 'https://raw.githubusercontent.com/tennessine/corpus/master/红楼梦.txt'
        print(f"正在下载 {url}...")
        try:
            with open(file_path, 'wb') as f:
                f.write(requests.get(url).content)
            print("下载完成。")
        except Exception as e:
            print(f"下载失败: {e}")
            raise
    else:
        print(f"{file_path} 已存在。")


class DataLoader:
    """数据加载器

    负责数据的加载、分词和批次生成。
    支持训练集/验证集分割。

    Args:
        file_path: 数据文件路径
        batch_size: 批量大小
        block_size: 序列长度(上下文窗口)
        device: 计算设备
        split_ratio: 训练集比例(默认 0.9)

    Attributes:
        tokenizer: 分词器实例
        tokens: 所有编码后的 token
        train_data: 训练集数据
        val_data: 验证集数据
    """

    def __init__(self, file_path='input_cn.txt', batch_size=32,
                 block_size=256, device='cpu', split_ratio=0.9):
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device

        # 1. 下载数据
        download_data(file_path)

        # 2. 初始化并训练 Tokenizer
        self.tokenizer = Tokenizer()
        if not os.path.exists("tokenizer.model"):
            # 如果没有训练好的 tokenizer,就现场训练一个
            # 注意:vocab_size 必须与配置中的一致
            self.tokenizer.train(file_path, vocab_size=8192)

        # 3. 读取文本
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 4. 编码
        print("正在编码数据...")
        self.tokens = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        print(f"数据加载完成。总 token 数: {len(self.tokens)}")

        # 5. 划分训练集/验证集
        n = int(split_ratio * len(self.tokens))
        self.train_data = self.tokens[:n]
        self.val_data = self.tokens[n:]
        print(f"训练集: {len(self.train_data)} tokens, 验证集: {len(self.val_data)} tokens")

    def get_batch(self, split='train'):
        """获取一个批次的数据

        Args:
            split: 'train' 或 'val'

        Returns:
            x: 输入序列,形状 (batch_size, block_size)
            y: 目标序列,形状 (batch_size, block_size)
        """
        data = self.train_data if split == 'train' else self.val_data

        # 生成随机起始位置
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))

        # 堆叠输入 (x) 和目标 (y)
        # y 只是 x 向后移动一位
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])

        # 移动到设备
        if self.device in ['cuda', 'mps']:
            x, y = x.to(self.device), y.to(self.device)

        return x, y

    def __len__(self):
        """返回数据集大小"""
        return len(self.tokens)
