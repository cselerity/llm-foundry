"""模型和训练配置

定义了模型架构和训练超参数的配置类。
"""

from dataclasses import dataclass

@dataclass
class ModelConfig:
    """模型架构配置

    Attributes:
        dim: Transformer 维度 (d_model)
        n_layers: Transformer 层数 (Block 数量)
        n_heads: 注意力头数
        n_kv_heads: KV 头数 (用于分组查询注意力 GQA)
        vocab_size: 词表大小
        max_seq_len: 最大序列长度(上下文窗口大小)
        dropout: Dropout 概率
    """
    dim: int = 256          # Transformer 维度 (d_model)
    n_layers: int = 4       # Transformer 层数 (Block 数量)
    n_heads: int = 8        # 注意力头数
    n_kv_heads: int = 4     # KV 头数 (用于分组查询注意力 GQA)
    vocab_size: int = 8192  # 词表大小
    max_seq_len: int = 256  # 上下文窗口大小
    dropout: float = 0.1    # Dropout 概率

@dataclass
class TrainConfig:
    """训练配置

    Attributes:
        batch_size: 批量大小
        learning_rate: 学习率
        max_iters: 最大迭代次数
        eval_interval: 评估间隔(步数)
        eval_iters: 每次评估的批次数
        device: 计算设备(将自动检测)
    """
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_iters: int = 1000    # 最大迭代次数(实际训练建议 5000+)
    eval_interval: int = 50  # 每隔多少步评估一次
    eval_iters: int = 20     # 每次评估使用多少个 batch
    device: str = 'cpu'      # 将在代码中自动检测
