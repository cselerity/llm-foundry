"""Transformer 语言模型

完整的 Transformer 架构实现,用于因果语言建模。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .components import Block, RMSNorm, precompute_freqs_cis


class MiniLLM(nn.Module):
    """Mini LLM: 轻量级 Transformer 语言模型

    这是一个完整的 decoder-only Transformer 模型,实现了现代 LLM 的关键特性:
    - Token Embedding
    - 多层 Transformer Blocks (带 RoPE、GQA、SwiGLU)
    - 输出投影到词表

    Args:
        cfg: 模型配置

    Attributes:
        token_embedding: Token 嵌入层
        layers: Transformer 层列表
        norm: 最终的层归一化
        output: 输出投影层(到词表)
        freqs_cis: 预计算的 RoPE 频率

    Shape:
        - Input: (batch, seq_len) - Token 索引
        - Output: (batch, seq_len, vocab_size) - Logits
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # 嵌入和输出层
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        # 权重绑定 (可选,但在许多 LLM 中很常见)
        # 共享 embedding 和 output 权重可以减少参数量
        # self.token_embedding.weight = self.output.weight

        # 预计算 RoPE 频率
        self.freqs_cis = precompute_freqs_cis(
            self.cfg.dim // self.cfg.n_heads,
            self.cfg.max_seq_len * 2  # *2 作为缓冲区
        )

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化模型权重

        使用正态分布初始化线性层和嵌入层的权重。
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, targets=None):
        """前向传播

        Args:
            tokens: 输入 token 索引,形状 (batch, seq_len)
            targets: 目标 token 索引(用于训练),形状 (batch, seq_len)

        Returns:
            logits: 输出 logits,形状 (batch, seq_len, vocab_size)
            loss: 如果提供了 targets,返回交叉熵损失,否则为 None
        """
        batch_size, seq_len = tokens.shape
        h = self.token_embedding(tokens)

        # 获取当前序列长度的 RoPE 频率
        freqs_cis = self.freqs_cis[:seq_len].to(h.device)

        # 通过所有 Transformer 层
        for layer in self.layers:
            h = layer(h, freqs_cis)

        # 最终归一化和输出投影
        h = self.norm(h)
        logits = self.output(h)

        loss = None
        if targets is not None:
            # 计算交叉熵损失
            # 展平以进行 CrossEntropyLoss 计算
            # logits: (B, S, V) -> (B*S, V)
            # targets: (B, S) -> (B*S)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def get_num_params(self, non_embedding=True):
        """获取模型参数数量

        Args:
            non_embedding: 是否排除嵌入层参数

        Returns:
            参数总数
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        return n_params
