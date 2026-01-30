"""SentencePiece 分词器

基于 SentencePiece 的 BPE (Byte Pair Encoding) 分词器实现。
"""

import os
import sentencepiece as spm


class Tokenizer:
    """SentencePiece BPE 分词器

    使用 SentencePiece 库实现的 BPE 分词器,适用于多语言文本。
    支持训练自定义词表和编码/解码功能。

    Args:
        model_path: 分词器模型文件路径

    Attributes:
        sp: SentencePiece 处理器实例
        model_path: 模型文件路径
    """

    def __init__(self, model_path="tokenizer.model"):
        self.model_path = model_path
        self.sp = spm.SentencePieceProcessor()
        if os.path.exists(model_path):
            self.load(model_path)

    def train(self, input_file, vocab_size=4096, model_prefix="tokenizer"):
        """在给定文本上训练 SentencePiece BPE 模型

        Args:
            input_file: 训练数据文件路径
            vocab_size: 词表大小
            model_prefix: 输出模型的前缀
        """
        print(f"正在训练 Tokenizer (vocab_size={vocab_size})...")
        # character_coverage=1.0 确保所有字符都被覆盖 (对于中文很重要)
        # model_type="bpe" 使用 Byte Pair Encoding
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            user_defined_symbols=['<pad>', '<bos>', '<eos>']
        )
        self.load(f"{model_prefix}.model")
        print("Tokenizer 训练完成。")

    def load(self, model_path):
        """加载训练好的分词器模型

        Args:
            model_path: 模型文件路径
        """
        self.sp.load(model_path)

    def encode(self, text):
        """将文本编码为 token ID 列表

        Args:
            text: 输入文本

        Returns:
            Token ID 列表
        """
        return self.sp.encode_as_ids(text)

    def decode(self, ids):
        """将 token ID 列表解码为文本

        Args:
            ids: Token ID 列表

        Returns:
            解码后的文本
        """
        return self.sp.decode_ids(ids)

    @property
    def vocab_size(self):
        """获取词表大小

        Returns:
            词表大小
        """
        return self.sp.get_piece_size()
