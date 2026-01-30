from dataclasses import dataclass

@dataclass
class ModelConfig:
    dim: int = 256          # Transformer 维度 (d_model)
    n_layers: int = 4       # Transformer 层数 (Block 数量)
    n_heads: int = 8        # 注意力头数
    n_kv_heads: int = 4     # KV 头数 (用于分组查询注意力 GQA，可选，可以与 n_heads 相同)
    vocab_size: int = 8192  # SentencePiece 词表大小 (增加以覆盖所有字符)
    max_seq_len: int = 256  # 上下文窗口大小
    dropout: float = 0.1
    
@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_iters: int = 1000    # 最大迭代次数 (为了演示设为 100，实际训练建议 5000+)
    eval_interval: int = 50 # 每隔多少步评估一次
    eval_iters: int = 20    # 每次评估使用多少个 batch
    device: str = 'cpu'     # 将在代码中自动检测
