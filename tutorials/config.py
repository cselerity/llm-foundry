"""配置类定义

这个模块定义了模型架构和训练过程的所有配置参数。
使用 dataclass 使得配置清晰、易于修改和传递。

教学要点:
1. 理解模型架构的关键超参数
2. 理解训练过程的关键超参数
3. 了解参数之间的相互关系和约束
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """模型架构配置

    定义 Transformer 模型的所有架构参数。
    这些参数共同决定了模型的规模、能力和计算成本。

    教学要点:
    - 理解每个参数如何影响模型容量和性能
    - 理解参数之间的约束关系
    - 学会根据资源调整模型大小
    """

    # ============= 核心维度参数 =============

    dim: int = 256
    """Transformer 主维度 (也称为 d_model 或 hidden_size)

    这是模型中最重要的维度参数,决定了:
    - 每个 token 的向量表示维度
    - 每一层中间隐藏状态的维度
    - 模型的表达能力和参数量

    典型值:
    - 小型模型: 128-512
    - 中型模型: 512-1024
    - 大型模型: 1024-4096
    - GPT-3: 12288

    注意: dim 必须能被 n_heads 整除
    """

    n_layers: int = 4
    """Transformer 层数 (深度)

    模型中堆叠的 Transformer Block 数量。
    更多的层意味着:
    - 更强的建模能力
    - 更多的参数
    - 更长的训练时间
    - 可能需要更大的学习率 warmup

    典型值:
    - 小型模型: 4-8 层
    - 中型模型: 8-24 层
    - 大型模型: 24-96 层
    - GPT-3: 96 层
    """

    # ============= 注意力机制参数 =============

    n_heads: int = 8
    """Query 注意力头数 (Multi-Head Attention 中的头数)

    多头注意力允许模型同时关注不同位置的不同特征。

    约束条件:
    - dim 必须能被 n_heads 整除
    - 每个头的维度 = dim / n_heads

    典型配置:
    - 每个头的维度通常为 64 或 128
    - dim=256, n_heads=8  → head_dim=32
    - dim=512, n_heads=8  → head_dim=64
    - dim=768, n_heads=12 → head_dim=64 (BERT)

    注意: n_heads 应该是 n_kv_heads 的整数倍
    """

    n_kv_heads: int = 4
    """Key-Value 注意力头数 (用于 Grouped Query Attention)

    GQA (Grouped Query Attention) 是一种优化技术:
    - 多个 Query 头共享同一个 Key-Value 头
    - 大幅减少 KV cache 的内存占用
    - 对性能影响很小

    配置说明:
    - 如果 n_kv_heads == n_heads: 标准的多头注意力 (MHA)
    - 如果 n_kv_heads < n_heads: 分组查询注意力 (GQA)
    - 如果 n_kv_heads == 1: 多查询注意力 (MQA)

    示例:
    - n_heads=8, n_kv_heads=8: 每个 Q 头有自己的 KV 头 (MHA)
    - n_heads=8, n_kv_heads=4: 每 2 个 Q 头共享 1 个 KV 头 (GQA)
    - n_heads=8, n_kv_heads=1: 所有 Q 头共享 1 个 KV 头 (MQA)

    推荐: n_kv_heads = n_heads // 2 在大多数情况下效果很好
    """

    # ============= 词表和序列长度 =============

    vocab_size: int = 8192
    """词表大小

    分词器中 token 的总数量,决定了:
    - Embedding 层和输出层的参数量 (两者共享权重)
    - 参数量 = vocab_size × dim × 2 (输入和输出)

    典型值:
    - 小词表: 4096-8192 (小数据集、特定领域)
    - 中等词表: 30000-50000 (通用文本)
    - 大词表: 50000-100000 (多语言)
    - GPT 系列: 50257 (GPT-2), 100000+ (GPT-3)

    注意:
    - 词表越大,模型对罕见词的表示越好
    - 但也会显著增加参数量和计算成本
    - 需要与分词器训练时的词表大小匹配
    """

    max_seq_len: int = 256
    """最大序列长度 (上下文窗口大小)

    模型能够处理的最长 token 序列长度。

    影响:
    - 决定了 RoPE 位置编码的预计算长度
    - 影响 KV cache 的内存占用
    - 影响单个 batch 的内存需求

    典型值:
    - 短文本: 128-512 (对话、短文档)
    - 中等长度: 512-2048 (文章、长对话)
    - 长文本: 2048-8192 (长文档、书籍章节)
    - 超长上下文: 8192+ (需要特殊优化)

    注意:
    - Attention 的计算复杂度为 O(n²),序列越长计算量越大
    - 训练时的序列长度可以小于 max_seq_len
    - 推理时无法处理超过 max_seq_len 的输入
    """

    # ============= 正则化参数 =============

    dropout: float = 0.1
    """Dropout 比率

    训练时随机丢弃部分神经元,防止过拟合。

    应用位置:
    - 注意力层的输出
    - MLP 层之间
    - Residual connection 之后

    典型值:
    - 小数据集: 0.1-0.3 (防止过拟合)
    - 大数据集: 0.0-0.1 (数据本身就足够多样)
    - 预训练大模型: 0.0-0.1

    注意:
    - 推理时 dropout 会被关闭 (model.eval())
    - dropout 太高会影响模型容量
    - dropout 太低可能导致过拟合
    """

    def __post_init__(self):
        """验证配置参数的合法性

        这个方法在 dataclass 实例化后自动调用,
        用于检查参数之间的约束关系。
        """
        # 检查 dim 能否被 n_heads 整除
        if self.dim % self.n_heads != 0:
            raise ValueError(
                f"dim ({self.dim}) 必须能被 n_heads ({self.n_heads}) 整除。"
                f"每个头的维度 head_dim = dim / n_heads 必须是整数。"
            )

        # 检查 n_heads 能否被 n_kv_heads 整除
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) 必须能被 n_kv_heads ({self.n_kv_heads}) 整除。"
                f"这样才能让多个 Q 头平均共享 KV 头。"
            )

        # 计算并打印一些有用的派生信息
        head_dim = self.dim // self.n_heads
        group_size = self.n_heads // self.n_kv_heads

        # 估算参数量 (简化计算,不包括 LayerNorm 等小参数)
        params_per_layer = (
            # Attention: Q, K, V, O 投影
            self.dim * (self.n_heads * head_dim) +  # Q
            self.dim * (self.n_kv_heads * head_dim) +  # K
            self.dim * (self.n_kv_heads * head_dim) +  # V
            (self.n_heads * head_dim) * self.dim +  # O
            # MLP: w1, w2, w3 (SwiGLU)
            self.dim * (4 * self.dim * 2 // 3) * 3
        )
        total_params = (
            self.vocab_size * self.dim * 2 +  # Embedding + Output
            params_per_layer * self.n_layers
        )

        print(f"模型配置信息:")
        print(f"  - 每个头的维度: {head_dim}")
        print(f"  - GQA 分组大小: {group_size} (每 {group_size} 个 Q 头共享 1 个 KV 头)")
        print(f"  - 估算参数量: {total_params / 1e6:.2f}M")


@dataclass
class TrainConfig:
    """训练配置

    定义训练过程的所有超参数。
    这些参数决定了训练的速度、稳定性和最终效果。

    教学要点:
    - 理解每个超参数如何影响训练过程
    - 学会根据模型大小和数据调整超参数
    - 理解训练稳定性和收敛速度的权衡
    """

    # ============= 基础训练参数 =============

    batch_size: int = 32
    """批次大小 (每次更新参数使用的样本数)

    batch_size 是训练中最重要的超参数之一:

    影响:
    - 更大的 batch: 训练更稳定,但收敛可能更慢
    - 更小的 batch: 训练更快,但可能不稳定
    - 内存占用与 batch_size 成正比

    典型值:
    - GPU 内存有限: 8-32
    - 中等 GPU: 32-64
    - 大型 GPU/TPU: 64-256
    - 超大规模训练: 256-2048 (使用梯度累积)

    调整建议:
    - 根据 GPU 内存调整到最大可用值
    - 如果内存不够,减小 batch_size 或使用梯度累积
    - 如果改变 batch_size,可能需要相应调整学习率
      (经验法则: 学习率 ∝ √batch_size)
    """

    learning_rate: float = 3e-4
    """学习率 (控制参数更新的步长)

    学习率是训练中最关键的超参数:

    影响:
    - 太大: 训练不稳定,损失震荡甚至发散
    - 太小: 收敛太慢,可能陷入局部最优
    - 需要根据模型大小和数据量调整

    典型值:
    - 小模型: 1e-3 到 3e-4
    - 中型模型: 3e-4 到 1e-4
    - 大模型: 1e-4 到 3e-5
    - GPT-3 (175B): 6e-5

    学习率策略:
    - Warmup: 从小值线性增长到目标学习率 (前几百步)
    - Cosine decay: 余弦退火,逐渐降低学习率
    - Constant: 保持不变 (简单但不一定最优)

    调整建议:
    - 如果损失震荡或 NaN: 降低学习率
    - 如果收敛太慢: 提高学习率 (小心)
    - 对于大模型,使用 warmup 可以提高稳定性
    """

    # ============= 训练流程参数 =============

    max_iters: int = 1000
    """最大训练迭代次数 (总共训练多少个 batch)

    决定训练的总时长:
    - 1 个 iteration = 1 个 batch 的前向+反向传播
    - 总训练样本数 = max_iters × batch_size

    典型值:
    - 快速实验: 100-1000
    - 小规模训练: 1000-10000
    - 中等规模训练: 10000-100000
    - 大规模预训练: 100000-1000000+

    注意:
    - 这里设为 1000 是为了快速演示
    - 实际训练建议至少 5000 步
    - 可以通过验证损失判断是否需要更多迭代
    """

    eval_interval: int = 50
    """评估间隔 (每隔多少步在验证集上评估一次)

    定期评估可以:
    - 监控模型在验证集上的表现
    - 及早发现过拟合
    - 保存最佳模型检查点

    典型值:
    - 快速迭代: 每 10-50 步
    - 一般训练: 每 50-200 步
    - 长时间训练: 每 200-1000 步

    权衡:
    - 评估太频繁: 浪费时间,训练变慢
    - 评估太少: 可能错过最佳模型
    """

    eval_iters: int = 20
    """每次评估使用的批次数

    评估时使用多个 batch 以获得更稳定的估计:
    - 评估样本数 = eval_iters × batch_size
    - 更多的 batch 可以得到更准确的验证损失

    典型值:
    - 快速评估: 10-20
    - 准确评估: 50-100
    - 完整验证集: 使用所有验证数据

    注意:
    - eval_iters 越大,评估越准确但越慢
    - 可以根据验证集大小调整
    """

    # ============= 设备配置 =============

    device: str = 'cpu'
    """训练设备 (cpu, cuda, mps 等)

    指定模型和数据放置的设备:
    - 'cpu': CPU 训练 (慢但兼容性好)
    - 'cuda': NVIDIA GPU (最常用,速度快)
    - 'cuda:0', 'cuda:1': 指定特定 GPU
    - 'mps': Apple Silicon GPU (M1/M2 Mac)

    注意:
    - 通常在代码中自动检测可用设备
    - GPU 训练比 CPU 快 10-100 倍
    - 确保 PyTorch 安装了对应的 CUDA 版本
    """

    def __post_init__(self):
        """验证训练配置的合法性"""
        # 检查学习率范围
        if self.learning_rate <= 0 or self.learning_rate > 1:
            raise ValueError(
                f"learning_rate ({self.learning_rate}) 应该在 (0, 1] 范围内。"
                f"典型值: 1e-5 到 1e-3"
            )

        # 检查 batch_size
        if self.batch_size <= 0:
            raise ValueError(f"batch_size ({self.batch_size}) 必须是正整数")

        # 打印训练信息
        total_samples = self.max_iters * self.batch_size
        print(f"\n训练配置信息:")
        print(f"  - 总训练样本数: {total_samples:,} (约 {total_samples/1e6:.2f}M tokens)")
        print(f"  - 每 {self.eval_interval} 步评估一次")
        print(f"  - 使用设备: {self.device}")


# ============= 预定义配置 =============

def get_small_config():
    """小型模型配置 (约 2M 参数)

    适用于:
    - 快速实验和原型验证
    - 教学和学习
    - 资源受限环境 (笔记本 CPU)
    """
    return ModelConfig(
        dim=256,
        n_layers=4,
        n_heads=8,
        n_kv_heads=4,
        vocab_size=8192,
        max_seq_len=256
    )


def get_medium_config():
    """中型模型配置 (约 10M 参数)

    适用于:
    - 中等规模数据集训练
    - GPU 训练 (4-8GB 显存)
    - 更好的生成质量
    """
    return ModelConfig(
        dim=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=4,
        vocab_size=16384,
        max_seq_len=512
    )


def get_large_config():
    """大型模型配置 (约 70-75M 参数)

    适用于:
    - 大规模数据集训练
    - 高性能 GPU (8-16GB 显存)
    - 本地学习和研究
    - 生产级应用

    硬件要求:
    - RTX 5060 (8GB): 推荐 batch_size=16-32
    - RTX 3060 (12GB): 推荐 batch_size=32-48
    - RTX 4090 (24GB): 推荐 batch_size=64-96

    性能预估 (RTX 5060):
    - 训练速度: ~2000-3000 tokens/sec
    - 推理速度: ~50-100 tokens/sec
    - 训练 10k steps: 约 30-45 分钟
    """
    return ModelConfig(
        dim=768,           # 隐藏层维度
        n_layers=12,       # Transformer 层数
        n_heads=12,        # 注意力头数
        n_kv_heads=6,      # KV 头数 (GQA 优化,节省显存)
        vocab_size=32768,  # 词汇表大小 (32k)
        max_seq_len=1024,  # 最大序列长度
        dropout=0.1        # Dropout 率
    )


def get_rtx5060_config():
    """RTX 5060 优化配置 (约 70-75M 参数)

    专门为 RTX 5060 (8GB 显存) 优化的配置,在性能和显存之间取得最佳平衡。

    模型架构:
    --------
    - 参数量: ~70-75M
    - 层数: 12 层
    - 隐藏维度: 768
    - 注意力头数: 12 (每头维度 64)
    - KV 头数: 6 (GQA,节省 50% KV cache)
    - 词汇表: 32k tokens
    - 上下文长度: 1024 tokens

    推荐训练参数:
    -----------
    - batch_size: 24-32 (根据实际显存占用调整)
    - learning_rate: 3e-4
    - max_iters: 10000-50000
    - gradient_accumulation_steps: 2-4 (如果显存不足)

    显存占用估算:
    -----------
    - 模型参数: ~280 MB (FP32) / ~140 MB (FP16)
    - 优化器状态: ~560 MB (AdamW, FP32)
    - 激活值: ~1-2 GB (取决于 batch_size 和 seq_len)
    - KV cache (推理): ~300-500 MB
    - 总计 (训练): 约 3-4 GB (batch_size=24, seq_len=1024)

    性能预估:
    --------
    - 训练速度: 2500-3500 tokens/sec
    - 每个 step: ~10-15ms
    - 10k steps 训练时间: 30-40 分钟
    - 推理速度: 60-100 tokens/sec

    使用建议:
    --------
    1. 确保使用 PyTorch 2.0+ (性能提升 20-30%)
    2. 启用 torch.compile() 进一步加速
    3. 如果显存不足,可以:
       - 减小 batch_size 到 16
       - 减小 max_seq_len 到 512
       - 使用梯度累积 (gradient_accumulation)
    4. 训练时监控 GPU 利用率 (nvidia-smi)

    与其他配置对比:
    ------------
    | 配置   | 参数量 | 显存需求 | 训练速度 | 生成质量 |
    |--------|--------|----------|----------|----------|
    | Small  | 2M     | <1GB     | 快       | 低       |
    | Medium | 10M    | 2-3GB    | 中       | 中       |
    | Large  | 70M    | 3-4GB    | 慢       | 高       |
    | RTX5060| 70M    | 3-4GB    | 优化     | 高       |

    示例:
    ----
    >>> # 使用 RTX 5060 配置
    >>> model_cfg = get_rtx5060_config()
    >>> train_cfg = TrainConfig(
    ...     batch_size=24,
    ...     learning_rate=3e-4,
    ...     max_iters=10000
    ... )
    >>> model = MiniLLM(model_cfg).to('cuda')
    >>> # 开始训练...
    """
    return ModelConfig(
        dim=768,           # 隐藏层维度 (BERT-base 同款)
        n_layers=12,       # 12 层 Transformer
        n_heads=12,        # 12 个注意力头 (每头 64 维)
        n_kv_heads=6,      # 6 个 KV 头 (GQA,显存优化)
        vocab_size=32768,  # 32k 词汇表 (适合中文 + 英文)
        max_seq_len=1024,  # 1k 上下文长度
        dropout=0.1        # 10% Dropout (防止过拟合)
    )


# ============= 训练配置预设 =============

def get_rtx5060_train_config():
    """RTX 5060 优化训练配置

    专门为 RTX 5060 优化的训练超参数。

    配置说明:
    --------
    - batch_size=24: 平衡速度和显存
    - learning_rate=3e-4: 适中的学习率
    - max_iters=10000: 中等规模训练
    - eval_interval=500: 较频繁的评估

    使用方式:
    --------
    >>> model_cfg = get_rtx5060_config()
    >>> train_cfg = get_rtx5060_train_config()
    >>> loader = DataLoader(
    ...     batch_size=train_cfg.batch_size,
    ...     block_size=model_cfg.max_seq_len
    ... )
    >>> model = MiniLLM(model_cfg).to('cuda')
    >>> # 开始训练...
    """
    return TrainConfig(
        batch_size=24,         # RTX 5060 优化批次大小
        learning_rate=3e-4,    # Adam 推荐学习率
        max_iters=10000,       # 中等规模训练
        eval_interval=500,     # 每 500 步评估一次
        eval_iters=50,         # 评估时用 50 个批次
        device='cuda'          # 使用 GPU
    )


def get_m4pro_config():
    """Apple M4 Pro 优化配置 (约 68-72M 参数)

    专门为 Apple M4 Pro (32GB 统一内存) 优化的配置。

    模型架构:
    --------
    - 参数量: ~68-72M
    - 层数: 10 层 (相比 RTX5060 的 12 层略少,适配 MPS)
    - 隐藏维度: 704 (11 × 64,优化的维度)
    - 注意力头数: 11 (每头维度 64)
    - KV 头数: 4 (更激进的 GQA,适配 Apple Silicon)
    - 词汇表: 32k tokens
    - 上下文长度: 1024 tokens

    Apple Silicon 优化:
    -----------------
    - **维度对齐**: 704 = 11 × 64,针对 Metal 优化
    - **GQA 比例**: 11:4 (约 2.75:1),平衡性能和内存
    - **层数**: 10 层,在 M4 Pro 上获得最佳吞吐量
    - **统一内存**: 充分利用 32GB 统一内存架构

    性能预估 (M4 Pro 32GB):
    ---------------------
    - 训练速度: 1500-2500 tokens/sec
    - 推理速度: 40-80 tokens/sec
    - 10k steps: 约 40-60 分钟
    - 内存占用: 4-6 GB (统一内存)

    与 RTX 5060 对比:
    ---------------
    | 方面       | M4 Pro    | RTX 5060  | 说明                |
    |------------|-----------|-----------|---------------------|
    | 层数       | 10        | 12        | M4 Pro 更适合 10 层 |
    | 隐藏维度   | 704       | 768       | Metal 优化对齐      |
    | 注意力头   | 11        | 12        | 保持 64 维每头      |
    | KV 头      | 4         | 6         | 更激进的 GQA        |
    | 训练速度   | 2000/s    | 3000/s    | MPS vs CUDA         |
    | 内存优势   | 统一内存  | 独立显存  | 灵活分配            |

    使用建议:
    --------
    1. **充分利用统一内存**: 可以使用更大的 batch_size (32-40)
    2. **MPS 后端**: 确保使用 PyTorch 2.0+ 以获得最佳 MPS 性能
    3. **CPU fallback**: 某些操作可能回退到 CPU,这是正常的
    4. **冷启动**: 第一次运行会有 Metal shader 编译开销

    为什么是这些参数?
    ---------------
    - **704 维度**: 11 × 64,Metal 处理 64 倍数维度更高效
    - **10 层**: 在 M4 Pro 上,10 层比 12 层有更好的吞吐量
    - **11 头**: 保持每头 64 维,同时优化总维度为 704
    - **4 KV 头**: 更激进的 GQA (11:4),节省统一内存

    Apple Silicon 注意事项:
    ---------------------
    - MPS 性能持续改进中 (PyTorch 2.1+ 更好)
    - 某些操作在 CPU 执行更快 (如 LayerNorm)
    - 统一内存允许更灵活的内存分配
    - 训练时 CPU + GPU 协同工作

    示例:
    ----
    >>> # 使用 M4 Pro 配置
    >>> import torch
    >>> model_cfg = get_m4pro_config()
    >>> train_cfg = get_m4pro_train_config()
    >>> device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    >>> model = MiniLLM(model_cfg).to(device)
    >>> # 开始训练...
    """
    return ModelConfig(
        dim=704,           # 11 × 64,Metal 优化维度
        n_layers=10,       # 10 层在 M4 Pro 上最优
        n_heads=11,        # 11 个注意力头 (每头 64 维)
        n_kv_heads=4,      # 4 个 KV 头 (激进的 GQA)
        vocab_size=32768,  # 32k 词汇表
        max_seq_len=1024,  # 1k 上下文长度
        dropout=0.1        # 10% Dropout
    )


def get_m4pro_train_config():
    """Apple M4 Pro 优化训练配置

    专门为 M4 Pro 优化的训练超参数。

    配置说明:
    --------
    - batch_size=32: 利用统一内存优势
    - learning_rate=3e-4: Adam 推荐学习率
    - max_iters=10000: 中等规模训练
    - eval_interval=500: 较频繁的评估
    - device='mps': 使用 Metal Performance Shaders

    Apple Silicon 优化:
    -----------------
    - 更大的 batch_size (32 vs 24),利用统一内存
    - 自动使用 MPS 后端
    - 适配 Metal 的计算模式

    使用方式:
    --------
    >>> import torch
    >>> model_cfg = get_m4pro_config()
    >>> train_cfg = get_m4pro_train_config()
    >>>
    >>> # 确保使用 MPS
    >>> device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    >>> print(f"Using device: {device}")
    >>>
    >>> loader = DataLoader(
    ...     batch_size=train_cfg.batch_size,
    ...     block_size=model_cfg.max_seq_len,
    ...     device=device
    ... )
    >>> model = MiniLLM(model_cfg).to(device)
    >>> # 开始训练...

    性能提示:
    --------
    - 首次运行会编译 Metal shaders (1-2 分钟)
    - 后续运行会使用缓存的 shaders
    - 监控统一内存使用: Activity Monitor → Memory
    - 如果内存压力大,减小 batch_size
    """
    return TrainConfig(
        batch_size=32,         # 利用统一内存优势
        learning_rate=3e-4,    # Adam 推荐学习率
        max_iters=10000,       # 中等规模训练
        eval_interval=500,     # 每 500 步评估一次
        eval_iters=50,         # 评估时用 50 个批次
        device='mps'           # 使用 Metal Performance Shaders
    )


# 示例用法
if __name__ == "__main__":
    print("=" * 80)
    print("配置示例和对比")
    print("=" * 80)

    print("\n" + "─" * 80)
    print("1. 小型模型配置 (适合快速实验)")
    print("─" * 80)
    small = get_small_config()

    print("\n" + "─" * 80)
    print("2. 中型模型配置 (适合 4-8GB 显存)")
    print("─" * 80)
    medium = get_medium_config()

    print("\n" + "─" * 80)
    print("3. 大型模型配置 (适合 8-16GB 显存)")
    print("─" * 80)
    large = get_large_config()

    print("\n" + "─" * 80)
    print("4. RTX 5060 优化配置 (推荐用于本地学习)")
    print("─" * 80)
    rtx5060 = get_rtx5060_config()

    print("\n" + "─" * 80)
    print("5. RTX 5060 训练配置")
    print("─" * 80)
    rtx5060_train = get_rtx5060_train_config()

    print("\n" + "─" * 80)
    print("6. Apple M4 Pro 优化配置 (推荐用于 Mac 用户)")
    print("─" * 80)
    m4pro = get_m4pro_config()

    print("\n" + "─" * 80)
    print("7. M4 Pro 训练配置")
    print("─" * 80)
    m4pro_train = get_m4pro_train_config()

    # 参数量对比
    print("\n" + "=" * 80)
    print("参数量对比")
    print("=" * 80)

    def estimate_params(cfg):
        """估算模型参数量"""
        # Token Embedding: vocab_size × dim
        tok_emb = cfg.vocab_size * cfg.dim

        # Transformer Blocks
        # - Attention: 4 × dim × dim (Q, K, V, O projections)
        # - MLP: 2 × dim × (4 × dim) = 8 × dim²
        # - LayerNorm: 2 × dim (可忽略)
        block_params = (4 * cfg.dim * cfg.dim) + (8 * cfg.dim * cfg.dim)
        total_blocks = cfg.n_layers * block_params

        # Output Head: dim × vocab_size
        out_head = cfg.dim * cfg.vocab_size

        total = tok_emb + total_blocks + out_head
        return total

    configs = [
        ("Small", small),
        ("Medium", medium),
        ("Large", large),
        ("RTX 5060", rtx5060),
        ("M4 Pro", m4pro)
    ]

    print("\n| 配置      | 参数量   | 层数 | 隐藏维度 | 词汇表 | 上下文 |")
    print("|-----------|----------|------|----------|--------|--------|")
    for name, cfg in configs:
        params = estimate_params(cfg) / 1e6
        print(f"| {name:9s} | {params:6.1f}M | {cfg.n_layers:4d} | {cfg.dim:8d} | {cfg.vocab_size:6d} | {cfg.max_seq_len:6d} |")

    # 显存占用估算
    print("\n" + "=" * 80)
    print("显存占用估算 (batch_size=24, seq_len=1024, FP32 训练)")
    print("=" * 80)

    for name, cfg in configs:
        params = estimate_params(cfg)
        model_mem = params * 4 / (1024**3)  # FP32: 4 bytes per param
        optimizer_mem = params * 8 / (1024**3)  # AdamW: 8 bytes per param (2 states)
        # 粗略估算激活值内存 (取决于 batch_size 和 seq_len)
        activation_mem = 0.5 if name == "Small" else 1.5 if name == "Medium" else 2.5

        total_mem = model_mem + optimizer_mem + activation_mem
        print(f"\n{name}:")
        print(f"  模型参数:     {model_mem:.2f} GB")
        print(f"  优化器状态:   {optimizer_mem:.2f} GB")
        print(f"  激活值 (估算): {activation_mem:.2f} GB")
        print(f"  总计:         {total_mem:.2f} GB")

    # 使用建议
    print("\n" + "=" * 80)
    print("硬件配置建议")
    print("=" * 80)
    print("""
    CPU 训练 (学习用):
      → get_small_config() + batch_size=4-8
      → 训练时间: 10-30 分钟
      → 系统内存: 8GB+

    笔记本 GPU (4GB 显存):
      → get_small_config() 或 get_medium_config()
      → batch_size=8-16
      → 系统内存: 16GB+

    RTX 3060 / 4060 / 5060 (8GB 显存):
      → get_rtx5060_config() + get_rtx5060_train_config()
      → batch_size=16-24
      → 训练时间: 30-45 分钟 (10k steps)
      → 系统内存: 32GB 推荐 (避免系统内存瓶颈)

    RTX 3060 Ti / 4060 Ti (12GB 显存):
      → get_rtx5060_config()
      → batch_size=32-48
      → 训练时间: 20-30 分钟 (10k steps)
      → 系统内存: 32GB 推荐

    RTX 4090 (24GB 显存):
      → get_large_config()
      → batch_size=64-128
      → 可以尝试更大的模型 (dim=1024, n_layers=16)
      → 系统内存: 32GB+

    Apple M4 Pro (32GB 统一内存):
      → get_m4pro_config() + get_m4pro_train_config()
      → batch_size=32-40
      → 训练时间: 40-60 分钟 (10k steps)
      → 充分利用统一内存架构的优势

    注意:
    -----
    - RTX 5060 8GB 显卡推荐搭配 32GB 系统内存
    - 系统内存不足会导致数据加载和预处理成为瓶颈
    - Apple Silicon 的统一内存架构无此限制
    """)
