import os
import requests
import torch
import numpy as np
from config import TrainConfig
from tokenizer import Tokenizer # 导入我们的自定义 Tokenizer

def download_data(file_path='input_cn.txt'):
    """如果文件不存在，则下载红楼梦数据集。"""
    if not os.path.exists(file_path):
        url = 'https://raw.githubusercontent.com/tennessine/corpus/master/红楼梦.txt'
        print(f"正在下载 {url}...")
        with open(file_path, 'wb') as f: 
            f.write(requests.get(url).content)
        print("下载完成。")
    else:
        print(f"{file_path} 已存在。")

class DataLoader:
    def __init__(self, file_path='input_cn.txt', batch_size=32, block_size=256, device='cpu'):
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        
        # 1. 下载数据
        download_data(file_path)
        
        # 2. 初始化并训练 Tokenizer
        self.tokenizer = Tokenizer()
        if not os.path.exists("tokenizer.model"):
            # 如果没有训练好的 tokenizer，就现场训练一个
            # 注意：vocab_size 必须与 config.py 中一致 (8192)
            self.tokenizer.train(file_path, vocab_size=8192)
            
        # 3. 读取文本
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # 4. 编码
        print("正在编码数据...")
        self.tokens = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        print(f"数据加载完成。总 token 数: {len(self.tokens)}")
        
        # 5. 划分训练集/验证集
        n = int(0.9 * len(self.tokens))
        self.train_data = self.tokens[:n]
        self.val_data = self.tokens[n:]
        
    def get_batch(self, split='train'):
        data = self.train_data if split == 'train' else self.val_data
        # 生成随机起始位置
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        
        # 堆叠输入 (x) 和目标 (y)
        # y 只是 x 向后移动一位
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        
        if self.device == 'cuda' or self.device == 'mps':
             x, y = x.to(self.device), y.to(self.device)
             
        return x, y

if __name__ == '__main__':
    # 测试数据加载器
    cfg = TrainConfig()
    loader = DataLoader(batch_size=4, block_size=8)
    x, y = loader.get_batch('train')
    
    print("\n--- Batch 检查 ---")
    print(f"输入形状: {x.shape}")
    print(f"目标形状: {y.shape}")
    
    print("\n--- Batch 中的第一个样本 ---")
    print(f"输入 (索引): {x[0].tolist()}")
    print(f"输入 (文本): {loader.tokenizer.decode(x[0].tolist())}")
    print(f"目标 (索引): {y[0].tolist()}")
    print(f"目标 (文本): {loader.tokenizer.decode(y[0].tolist())}")
