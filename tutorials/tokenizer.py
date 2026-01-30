"""
Tokenizer - 文本分词器
====================

本模块基于 SentencePiece 实现了一个简单易用的分词器,用于将文本转换为 Token ID 序列。

教学目标:
--------
- 理解分词 (Tokenization) 的作用
- 掌握 SentencePiece BPE 的使用
- 学习词汇表和 Token ID 的概念
- 理解字符覆盖率和特殊 token

核心概念:
--------
1. **Tokenization (分词)**:
   将连续的文本切分为离散的 token 单元
   例如: "红楼梦" → ["红", "楼", "梦"] 或 ["红楼", "梦"]

2. **Byte Pair Encoding (BPE)**:
   一种子词 (subword) 分词算法
   - 从字符开始,逐步合并高频词对
   - 平衡词汇表大小和表示能力
   - 可以处理未见过的词 (通过子词组合)

3. **SentencePiece**:
   Google 开源的分词工具
   - 语言无关 (支持中文、英文、代码等)
   - 端到端训练 (无需预分词)
   - 直接处理原始文本

为什么需要分词?
-------------
神经网络无法直接处理文本字符串,需要将其转换为数字序列。
分词是这个转换过程的第一步:

    原始文本
        ↓ Tokenization
    Token 序列 (字符串)
        ↓ Encoding
    Token ID 序列 (整数)
        ↓ Embedding
    向量序列 (浮点数)
        ↓ Model
    输出向量
        ↓ Decoding
    预测文本

BPE vs 其他分词方法:
------------------
| 方法          | 词汇表大小 | 灵活性 | 适用场景               |
|---------------|-----------|--------|------------------------|
| 字符级 (Char) | 很小 (~100) | 低     | 简单任务,序列很长     |
| 词级 (Word)   | 很大 (~50k) | 低     | 词汇固定的语言        |
| **BPE/Subword** | 适中 (~8k)  | **高** | **多数现代 LLM (推荐)** |

设计哲学:
--------
- **简洁性**: 最小化 API,专注核心功能
- **易用性**: 自动处理训练和加载
- **通用性**: 支持中文、英文、代码等
- **标准化**: 使用业界标准 SentencePiece

与主工程的关系:
-------------
本文件是 src/llm_foundry/tokenizers/ 的教学展示版本。
主工程支持更多分词器类型,但 SentencePiece 是核心实现。
"""

import os
import sentencepiece as spm


# ============================================================================
# Tokenizer 类
# ============================================================================

class Tokenizer:
    """基于 SentencePiece 的分词器

    功能:
    ----
    1. 训练分词器: 从文本数据学习词汇表
    2. 加载分词器: 加载已训练的模型
    3. 编码: 文本 → Token ID 序列
    4. 解码: Token ID 序列 → 文本

    使用流程:
    --------
        初始化 Tokenizer
            ↓
        训练 (如果模型不存在)
            ↓
        编码文本为 IDs
            ↓
        ... 模型训练/推理 ...
            ↓
        解码 IDs 为文本

    参数:
    ----
    model_path : str
        分词器模型文件路径,默认 "tokenizer.model"

    属性:
    ----
    sp : SentencePieceProcessor
        底层 SentencePiece 处理器
    vocab_size : int
        词汇表大小 (通过 @property 获取)

    使用示例:
    --------
    >>> # 训练新分词器
    >>> tokenizer = Tokenizer()
    >>> tokenizer.train("data.txt", vocab_size=8192)
    >>>
    >>> # 编码文本
    >>> ids = tokenizer.encode("红楼梦是一部经典小说")
    >>> print(ids)  # [1234, 5678, 9012, ...]
    >>>
    >>> # 解码回文本
    >>> text = tokenizer.decode(ids)
    >>> print(text)  # "红楼梦是一部经典小说"
    """

    def __init__(self, model_path="tokenizer.model"):
        """初始化分词器

        教学要点:
        -------
        - 如果模型文件存在,自动加载
        - 如果不存在,需要先调用 train() 方法
        - 一个分词器可以用于多个数据集 (只要语言相同)

        参数:
        ----
        model_path : str
            分词器模型文件路径

        自动加载机制:
        -----------
        如果 model_path 指向的文件存在,构造函数会自动加载它。
        这样避免了每次使用都要手动调用 load()。

        注意事项:
        --------
        - SentencePiece 会同时生成 .model 和 .vocab 文件
        - .model 是训练好的模型 (必需)
        - .vocab 是人类可读的词汇表 (可选,用于调试)
        """
        self.model_path = model_path
        self.sp = spm.SentencePieceProcessor()
        if os.path.exists(model_path):
            self.load(model_path)

    def train(self, input_file, vocab_size=4096, model_prefix="tokenizer"):
        """在给定文本上训练 SentencePiece BPE 模型

        教学要点:
        -------
        - 训练时间: 通常几秒到几分钟 (取决于数据大小)
        - 只需训练一次,模型保存后可重复使用
        - vocab_size 是超参数,需要权衡

        参数:
        ----
        input_file : str
            训练数据文件路径 (原始文本)
        vocab_size : int
            词汇表大小,默认 4096
            - 更大: 表达能力强,但模型参数多
            - 更小: 模型紧凑,但序列变长
        model_prefix : str
            输出模型的文件名前缀,默认 "tokenizer"
            - 会生成 tokenizer.model 和 tokenizer.vocab

        训练参数详解:
        -----------
        - **model_type="bpe"**: 使用 Byte Pair Encoding
          其他选项: unigram, char, word
          BPE 是多数现代 LLM 的选择 (GPT, LLaMA 等)

        - **character_coverage=1.0**: 字符覆盖率 100%
          确保所有字符都被包含在词汇表中
          对于中文非常重要 (中文字符数量巨大)
          英文通常可以用 0.9995 (忽略极罕见字符)

        - **user_defined_symbols=['<pad>', '<bos>', '<eos>']**:
          用户自定义特殊 token
          - <pad>: padding token (序列对齐)
          - <bos>: begin of sequence (序列开始)
          - <eos>: end of sequence (序列结束)
          注意: 本教学实现中实际未使用这些特殊 token

        vocab_size 选择指南:
        -------------------
        | 数据类型 | 推荐大小 | 理由                    |
        |----------|---------|-------------------------|
        | 中文文学 | 8k-16k  | 汉字多,需要较大词汇表   |
        | 英文文本 | 4k-8k   | 词汇相对固定            |
        | 代码     | 8k-32k  | 变量名、API 等多样性高  |
        | 多语言   | 16k-32k | 覆盖多种语言的字符集    |

        训练过程:
        --------
        1. 统计字符频率
        2. 初始化词汇表为单字符
        3. 迭代合并高频字符对
        4. 直到词汇表达到 vocab_size
        5. 保存模型文件

        示例:
        ----
        >>> tokenizer = Tokenizer()
        >>> tokenizer.train("input.txt", vocab_size=8192)
        正在训练 Tokenizer (vocab_size=8192)...
        Tokenizer 训练完成。
        """
        print(f"正在训练 Tokenizer (vocab_size={vocab_size})...")

        # 调用 SentencePiece 训练器
        # 这是一个耗时操作,输出会显示训练进度
        spm.SentencePieceTrainer.train(
            input=input_file,             # 训练数据文件
            model_prefix=model_prefix,    # 输出模型前缀
            vocab_size=vocab_size,        # 词汇表大小
            model_type="bpe",             # 使用 BPE 算法
            character_coverage=1.0,       # 100% 字符覆盖 (中文必需)
            user_defined_symbols=['<pad>', '<bos>', '<eos>']  # 特殊 tokens
        )

        # 训练完成后自动加载模型
        self.load(f"{model_prefix}.model")
        print("Tokenizer 训练完成。")

    def load(self, model_path):
        """加载已训练的分词器模型

        教学要点:
        -------
        - 加载速度很快 (通常 < 1 秒)
        - 模型文件通常很小 (几 MB)
        - 可以多次加载不同的模型

        参数:
        ----
        model_path : str
            模型文件路径 (通常是 .model 文件)

        使用场景:
        --------
        - 初始化时自动加载 (如果文件存在)
        - 切换不同的分词器
        - 从检查点恢复训练

        注意事项:
        --------
        - 必须使用训练时生成的 .model 文件
        - 模型文件包含了完整的词汇表和合并规则
        - 不需要同时加载 .vocab 文件 (那只是人类可读版本)
        """
        self.sp.load(model_path)

    def encode(self, text):
        """将文本编码为 Token ID 序列

        教学要点:
        -------
        - 这是将文本转换为模型可处理格式的关键步骤
        - 返回的是整数列表 (Python list)
        - 每个整数范围: [0, vocab_size)

        参数:
        ----
        text : str
            原始文本字符串

        返回:
        ----
        list of int
            Token ID 序列

        编码过程:
        --------
        1. 文本 → 可能的分词方案
        2. 根据 BPE 规则选择最优分词
        3. 每个 token 映射到对应的 ID

        示例:
        ----
        >>> tokenizer.encode("红楼梦")
        [1234, 5678, 9012]

        >>> tokenizer.encode("This is a test")
        [123, 456, 78, 901]

        性能考虑:
        --------
        - 编码速度: 约 100k tokens/秒 (取决于 CPU)
        - 内存占用: 输出列表大小 = 序列长度 × 8 bytes
        - 对于大文本,考虑批量处理
        """
        return self.sp.encode_as_ids(text)

    def decode(self, ids):
        """将 Token ID 序列解码为文本

        教学要点:
        -------
        - 编码的逆操作
        - 可以完美恢复原始文本 (如果分词器训练得当)
        - 主要用于生成文本和调试

        参数:
        ----
        ids : list of int
            Token ID 序列

        返回:
        ----
        str
            解码后的文本字符串

        解码过程:
        --------
        1. 每个 ID 映射到对应的 token (字符串)
        2. 连接所有 tokens
        3. 移除特殊标记 (如 ▁ 表示空格)

        示例:
        ----
        >>> ids = [1234, 5678, 9012]
        >>> tokenizer.decode(ids)
        "红楼梦"

        >>> ids = [123, 456, 78, 901]
        >>> tokenizer.decode(ids)
        "This is a test"

        注意事项:
        --------
        - 输入必须是有效的 ID (0 到 vocab_size-1)
        - 无效 ID 会被替换为 <unk> (unknown token)
        - 解码后的文本可能与原文有微小差异 (空格等)

        使用场景:
        --------
        - 文本生成后转换为人类可读格式
        - 调试分词质量
        - 验证编码/解码的正确性
        """
        return self.sp.decode_ids(ids)

    @property
    def vocab_size(self):
        """获取词汇表大小

        教学要点:
        -------
        - 这是一个只读属性 (使用 @property 装饰器)
        - 词汇表大小在训练时确定,之后不可更改
        - 模型的 embedding 层和输出层必须与 vocab_size 匹配

        返回:
        ----
        int
            词汇表大小 (训练时指定的 vocab_size)

        使用场景:
        --------
        - 初始化模型时设置 embedding 层大小
        - 验证模型配置是否与分词器匹配
        - 计算模型参数量

        示例:
        ----
        >>> tokenizer = Tokenizer()
        >>> tokenizer.train("data.txt", vocab_size=8192)
        >>> print(tokenizer.vocab_size)
        8192

        >>> # 模型初始化时使用
        >>> embedding = nn.Embedding(tokenizer.vocab_size, embed_dim)
        """
        return self.sp.get_piece_size()


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == '__main__':
    """测试和演示分词器

    运行方式:
    --------
    python tokenizer.py

    预期输出:
    --------
    1. 训练分词器 (如果需要)
    2. 编码/解码示例
    3. 词汇表统计
    4. 分词质量检查

    教学价值:
    --------
    - 验证分词器基本功能
    - 观察不同文本的分词结果
    - 理解编码和解码的对应关系
    - 检查中文、英文、数字的处理
    """
    print("=" * 60)
    print("Tokenizer 测试")
    print("=" * 60)

    # ========================================================================
    # 测试 1: 初始化和训练 (如果需要)
    # ========================================================================
    print("\n" + "─" * 60)
    print("测试 1: 初始化分词器")
    print("─" * 60)

    tokenizer = Tokenizer()

    # 如果没有训练好的模型,需要先准备数据
    if not os.path.exists("tokenizer.model"):
        print("未找到 tokenizer.model")
        print("请先运行 dataloader.py 或准备训练数据")
        print("示例: tokenizer.train('input_cn.txt', vocab_size=8192)")
    else:
        print(f"✓ 成功加载分词器")
        print(f"  词汇表大小: {tokenizer.vocab_size}")

    # ========================================================================
    # 测试 2: 中文编码和解码
    # ========================================================================
    print("\n" + "─" * 60)
    print("测试 2: 中文文本编码/解码")
    print("─" * 60)

    text_cn = "红楼梦是中国古典四大名著之一"
    if os.path.exists("tokenizer.model"):
        ids_cn = tokenizer.encode(text_cn)
        decoded_cn = tokenizer.decode(ids_cn)

        print(f"原始文本: {text_cn}")
        print(f"Token IDs: {ids_cn}")
        print(f"Token 数量: {len(ids_cn)}")
        print(f"解码文本: {decoded_cn}")
        print(f"是否一致: {'✓ 是' if text_cn == decoded_cn else '✗ 否'}")

    # ========================================================================
    # 测试 3: 英文编码和解码
    # ========================================================================
    print("\n" + "─" * 60)
    print("测试 3: 英文文本编码/解码")
    print("─" * 60)

    text_en = "This is a simple test of the tokenizer."
    if os.path.exists("tokenizer.model"):
        ids_en = tokenizer.encode(text_en)
        decoded_en = tokenizer.decode(ids_en)

        print(f"原始文本: {text_en}")
        print(f"Token IDs: {ids_en}")
        print(f"Token 数量: {len(ids_en)}")
        print(f"解码文本: {decoded_en}")
        print(f"是否一致: {'✓ 是' if text_en == decoded_en else '✗ 否'}")

    # ========================================================================
    # 测试 4: 混合文本编码
    # ========================================================================
    print("\n" + "─" * 60)
    print("测试 4: 中英混合 + 数字")
    print("─" * 60)

    text_mixed = "GPT-3 有 175B 参数，是当时最大的模型。"
    if os.path.exists("tokenizer.model"):
        ids_mixed = tokenizer.encode(text_mixed)
        decoded_mixed = tokenizer.decode(ids_mixed)

        print(f"原始文本: {text_mixed}")
        print(f"Token IDs: {ids_mixed}")
        print(f"Token 数量: {len(ids_mixed)}")
        print(f"解码文本: {decoded_mixed}")

    # ========================================================================
    # 测试 5: 分词粒度检查
    # ========================================================================
    print("\n" + "─" * 60)
    print("测试 5: 分词粒度观察")
    print("─" * 60)

    if os.path.exists("tokenizer.model"):
        test_texts = [
            "人工智能",       # 常见词,可能作为整体
            "abcdefghijk",    # 未见过的词,会被切分
            "123456",         # 数字
            "Hello, World!",  # 英文 + 标点
        ]

        for text in test_texts:
            ids = tokenizer.encode(text)
            print(f"'{text}' → {len(ids)} tokens → {ids}")

    # ========================================================================
    # 测试 6: 词汇表统计
    # ========================================================================
    print("\n" + "─" * 60)
    print("测试 6: 词汇表信息")
    print("─" * 60)

    if os.path.exists("tokenizer.model"):
        print(f"词汇表大小:     {tokenizer.vocab_size}")
        print(f"模型文件路径:   {tokenizer.model_path}")
        print(f"模型文件大小:   {os.path.getsize(tokenizer.model_path) / 1024:.1f} KB")

        # 检查词汇表文件
        vocab_path = tokenizer.model_path.replace('.model', '.vocab')
        if os.path.exists(vocab_path):
            print(f"词汇表文件:     {vocab_path}")
            print(f"  (可以打开查看人类可读的词汇表)")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
