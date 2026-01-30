"""
DataLoader - 数据加载和批次生成
===============================

本模块实现了语言模型训练所需的完整数据流水线,从原始文本到可训练的批次数据。

教学目标:
--------
- 理解语言模型的数据准备流程
- 掌握 Token 序列的批次生成
- 学习训练集/验证集划分
- 理解因果语言模型的输入输出关系

核心概念:
--------
1. **数据下载**: 自动获取训练数据
2. **分词 (Tokenization)**: 文本 → Token ID 序列
3. **数据划分**: 训练集 90% + 验证集 10%
4. **批次生成**: 随机采样固定长度的序列
5. **输入输出对齐**: y = x 向后移动一位 (next token prediction)

数据流程:
--------
    原始文本 (红楼梦.txt)
        ↓ download_data()
    本地文件
        ↓ Tokenizer.encode()
    Token ID 序列 [101, 234, 567, ...]
        ↓ train/val split
    训练集 (90%) + 验证集 (10%)
        ↓ get_batch()
    随机批次 (batch_size, block_size)

设计哲学:
--------
- **简洁性**: 最小化依赖,专注核心功能
- **教育性**: 清晰展示每个步骤
- **实用性**: 可直接用于训练
- **可扩展**: 易于替换其他数据源

与主工程的关系:
-------------
本文件是 src/llm_foundry/data/ 的教学展示版本,功能完全对等。
主工程版本支持更多数据格式和高级功能,但核心逻辑相同。
"""

import os
import requests
import torch
import numpy as np
from config import TrainConfig
from tokenizer import Tokenizer  # 导入我们的自定义 Tokenizer


# ============================================================================
# 数据下载工具
# ============================================================================

def download_data(file_path='input_cn.txt'):
    """下载训练数据 (如果本地不存在)

    教学要点:
    -------
    - 自动化数据准备流程
    - 避免重复下载
    - 使用 UTF-8 编码处理中文

    参数:
    ----
    file_path : str
        本地文件路径,默认 'input_cn.txt'

    示例数据:
    --------
    - 红楼梦: 经典中文小说,约 70 万字
    - 适合测试中文语言模型
    - Token 数量约 100 万个 (取决于分词器)

    扩展建议:
    --------
    - 替换为其他数据源 (英文、代码等)
    - 支持多文件数据集
    - 添加数据清洗步骤
    """
    if not os.path.exists(file_path):
        url = 'https://raw.githubusercontent.com/tennessine/corpus/master/红楼梦.txt'
        print(f"正在下载 {url}...")
        with open(file_path, 'wb') as f:
            f.write(requests.get(url).content)
        print("下载完成。")
    else:
        print(f"{file_path} 已存在。")


# ============================================================================
# 数据加载器
# ============================================================================

class DataLoader:
    """语言模型的数据加载器

    功能:
    ----
    1. 自动下载和准备数据
    2. 训练/加载分词器
    3. 将文本编码为 Token ID 序列
    4. 划分训练集和验证集
    5. 生成随机批次用于训练

    工作流程:
    --------
        初始化 (__init__)
            ↓
        下载数据 (如果需要)
            ↓
        训练/加载 Tokenizer
            ↓
        文本编码为 Token IDs
            ↓
        划分 train/val (90%/10%)
            ↓
        随时可调用 get_batch()

    参数:
    ----
    file_path : str
        数据文件路径,默认 'input_cn.txt'
    batch_size : int
        批次大小,即一个 batch 包含多少个序列
        典型值: 8-64 (取决于 GPU 内存)
    block_size : int
        序列长度,即每个序列包含多少个 token
        也称 context_length 或 max_seq_len
        典型值: 128-2048
    device : str
        计算设备 ('cpu', 'cuda', 'mps')

    属性:
    ----
    tokenizer : Tokenizer
        分词器实例
    tokens : torch.Tensor
        完整的 Token ID 序列
    train_data : torch.Tensor
        训练集 (90%)
    val_data : torch.Tensor
        验证集 (10%)

    使用示例:
    --------
    >>> loader = DataLoader(batch_size=32, block_size=256)
    >>> x, y = loader.get_batch('train')
    >>> print(x.shape)  # torch.Size([32, 256])
    >>> print(y.shape)  # torch.Size([32, 256])
    """

    def __init__(self, file_path='input_cn.txt', batch_size=32, block_size=256, device='cpu'):
        """初始化数据加载器

        教学要点:
        -------
        - 数据准备的完整流程
        - 分词器训练只需执行一次
        - 训练/验证集划分比例 (90%/10%)
        - 所有数据加载到内存 (适合中小型数据集)

        步骤详解:
        -------
        1. 下载数据: 自动获取红楼梦文本
        2. 训练分词器: 如果不存在 tokenizer.model
           - 使用 SentencePiece BPE
           - vocab_size = 8192 (必须与 config.py 一致!)
        3. 文本编码: 字符串 → Token ID 列表
        4. 张量转换: Python list → torch.Tensor
        5. 数据划分: 前 90% 训练,后 10% 验证

        为什么是 90%/10%?
        ---------------
        - 标准的训练/验证划分比例
        - 验证集足够大以评估泛化性能
        - 不需要测试集 (这是教学项目)

        性能考虑:
        --------
        - 所有数据加载到内存: 快速但占用内存
        - 对于大型数据集,考虑使用 memory-mapped files
        - Token 序列作为 torch.long 存储 (4 bytes per token)
        """
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device

        # 步骤 1: 下载数据
        # ----------------
        # 如果本地没有数据文件,自动从 GitHub 下载
        download_data(file_path)

        # 步骤 2: 初始化并训练 Tokenizer
        # -----------------------------
        self.tokenizer = Tokenizer()
        if not os.path.exists("tokenizer.model"):
            # 如果没有训练好的 tokenizer,现场训练一个
            # 注意: vocab_size 必须与 config.py 中一致 (8192)
            # 训练时间: 约 10-30 秒 (取决于数据大小和 CPU)
            print("未找到 tokenizer.model,开始训练分词器...")
            self.tokenizer.train(file_path, vocab_size=8192)
        else:
            print("加载已有的 tokenizer.model")

        # 步骤 3: 读取文本
        # ---------------
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 步骤 4: 编码为 Token IDs
        # -----------------------
        # 文本 → Token ID 列表 → torch.Tensor
        # 例如: "红楼梦" → [1234, 5678, 9012] → tensor([1234, 5678, 9012])
        print("正在编码数据...")
        self.tokens = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        print(f"数据加载完成。总 token 数: {len(self.tokens)}")

        # 步骤 5: 划分训练集/验证集
        # ------------------------
        # 训练集: 前 90% 的数据
        # 验证集: 后 10% 的数据
        # 注意: 这是顺序划分,不是随机划分
        # 对于语言模型通常没问题,因为训练时会随机采样
        n = int(0.9 * len(self.tokens))
        self.train_data = self.tokens[:n]
        self.val_data = self.tokens[n:]

        print(f"训练集大小: {len(self.train_data)} tokens")
        print(f"验证集大小: {len(self.val_data)} tokens")
        
    def get_batch(self, split='train'):
        """生成一个随机批次

        教学要点:
        -------
        - 随机采样提高训练效果
        - 输入和目标错位一个 token
        - 批次张量形状: (batch_size, block_size)

        参数:
        ----
        split : str
            'train' 或 'val',选择训练集或验证集

        返回:
        ----
        x : torch.Tensor
            输入序列,形状 (batch_size, block_size)
        y : torch.Tensor
            目标序列,形状 (batch_size, block_size)

        核心逻辑:
        --------
        1. **随机起始位置**:
           - 从数据中随机选择 batch_size 个起始位置
           - 每个位置可以提取一个长度为 block_size 的序列
           - 确保不会超出数据范围: len(data) - block_size

        2. **输入和目标对齐**:
           - x = data[i : i+block_size]
           - y = data[i+1 : i+block_size+1]
           - y 是 x 向后移动一位

        为什么 y = x 向后移动一位?
        -------------------------
        这是因果语言模型 (Causal Language Model) 的核心设计:
        - 给定前面的 tokens,预测下一个 token
        - x[0] → y[0] (预测第 1 个 token)
        - x[1] → y[1] (预测第 2 个 token)
        - ...
        - x[n-1] → y[n-1] (预测第 n 个 token)

        举例:
        ----
        假设数据是: "红楼梦是一部经典小说"
        Token IDs:  [101, 202, 303, 404, 505, 606, 707, 808]

        如果 block_size = 4, batch_size = 2, 随机起始位置 ix = [0, 3]

        Batch 1 (起始位置 0):
            x = [101, 202, 303, 404]  "红楼梦是"
            y = [202, 303, 404, 505]  "楼梦是一"

        Batch 2 (起始位置 3):
            x = [404, 505, 606, 707]  "是一部经"
            y = [505, 606, 707, 808]  "一部经典"

        最终返回:
            x.shape = (2, 4)  # 2 个序列,每个长度 4
            y.shape = (2, 4)

        训练时的使用:
        -----------
        >>> x, y = loader.get_batch('train')
        >>> logits, loss = model(x, y)  # 前向传播
        >>> loss.backward()              # 反向传播
        >>> optimizer.step()             # 更新参数

        性能考虑:
        --------
        - 随机采样: O(batch_size) 时间
        - 数据移动到 GPU: 如果 device != 'cpu'
        - 每次调用返回不同的批次 (因为随机采样)
        """
        # 选择训练集或验证集
        data = self.train_data if split == 'train' else self.val_data

        # 生成随机起始位置
        # torch.randint(high, size) 返回 [0, high) 范围内的随机整数
        # 这里 high = len(data) - block_size, 确保可以提取完整序列
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))

        # 堆叠输入 (x) 和目标 (y)
        # x: 从每个起始位置提取 block_size 个 tokens
        # y: 从每个起始位置+1 提取 block_size 个 tokens
        # torch.stack() 将列表中的张量沿新维度堆叠
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])

        # 如果使用 GPU,将数据移动到 GPU
        if self.device == 'cuda' or self.device == 'mps':
            x, y = x.to(self.device), y.to(self.device)

        return x, y


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == '__main__':
    """测试和演示数据加载器

    运行方式:
    --------
    python dataloader.py

    预期输出:
    --------
    1. 数据下载信息 (如果需要)
    2. 分词器训练信息 (如果需要)
    3. 数据编码和划分信息
    4. Batch 形状和内容示例

    教学价值:
    --------
    - 验证数据加载流程
    - 观察输入和目标的对齐关系
    - 理解批次生成的随机性
    - 检查分词器的编码/解码质量
    """
    print("=" * 60)
    print("DataLoader 测试")
    print("=" * 60)

    # 初始化配置和数据加载器
    # batch_size=4: 小批次方便观察
    # block_size=8: 短序列便于展示
    cfg = TrainConfig()
    loader = DataLoader(batch_size=4, block_size=8)

    # 生成一个训练批次
    x, y = loader.get_batch('train')

    # ========================================================================
    # 检查 1: Batch 形状
    # ========================================================================
    print("\n" + "─" * 60)
    print("检查 1: Batch 形状")
    print("─" * 60)
    print(f"输入形状: {x.shape}  # 应该是 (batch_size=4, block_size=8)")
    print(f"目标形状: {y.shape}  # 应该和输入形状相同")

    # ========================================================================
    # 检查 2: 第一个样本的详细内容
    # ========================================================================
    print("\n" + "─" * 60)
    print("检查 2: 第一个样本 (输入 vs 目标)")
    print("─" * 60)
    print(f"输入 Token IDs: {x[0].tolist()}")
    print(f"输入文本内容: '{loader.tokenizer.decode(x[0].tolist())}'")
    print()
    print(f"目标 Token IDs: {y[0].tolist()}")
    print(f"目标文本内容: '{loader.tokenizer.decode(y[0].tolist())}'")
    print()
    print("注意: 目标是输入向后移动一位")
    print("     输入的最后一个 token 对应目标的第一个 token 的下一个")

    # ========================================================================
    # 检查 3: 验证输入输出对齐
    # ========================================================================
    print("\n" + "─" * 60)
    print("检查 3: 逐 Token 对齐验证")
    print("─" * 60)
    print("位置  输入ID  →  目标ID  (输入文本 → 目标文本)")
    print("─" * 60)
    for i in range(min(5, len(x[0]))):  # 只显示前 5 个 token
        input_token = x[0][i].item()
        target_token = y[0][i].item()
        input_text = loader.tokenizer.decode([input_token])
        target_text = loader.tokenizer.decode([target_token])
        print(f"[{i}]   {input_token:5d}  →  {target_token:5d}  ('{input_text}' → '{target_text}')")

    # ========================================================================
    # 检查 4: 数据统计信息
    # ========================================================================
    print("\n" + "─" * 60)
    print("检查 4: 数据集统计")
    print("─" * 60)
    print(f"总 Token 数:    {len(loader.tokens):,}")
    print(f"训练集 Tokens:  {len(loader.train_data):,} ({len(loader.train_data)/len(loader.tokens)*100:.1f}%)")
    print(f"验证集 Tokens:  {len(loader.val_data):,} ({len(loader.val_data)/len(loader.tokens)*100:.1f}%)")
    print(f"词汇表大小:     {loader.tokenizer.vocab_size}")

    # ========================================================================
    # 检查 5: 批次随机性
    # ========================================================================
    print("\n" + "─" * 60)
    print("检查 5: 批次随机性验证")
    print("─" * 60)
    print("生成 3 个连续批次,验证每次都不同:")
    for i in range(3):
        x_test, _ = loader.get_batch('train')
        print(f"  Batch {i+1} 第一个样本的前 3 个 tokens: {x_test[0][:3].tolist()}")

    print("\n" + "=" * 60)
    print("测试完成! 数据加载器工作正常。")
    print("=" * 60)
