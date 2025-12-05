import os
import sentencepiece as spm

class Tokenizer:
    def __init__(self, model_path="tokenizer.model"):
        self.model_path = model_path
        self.sp = spm.SentencePieceProcessor()
        if os.path.exists(model_path):
            self.load(model_path)
            
    def train(self, input_file, vocab_size=4096, model_prefix="tokenizer"):
        """
        在给定文本上训练 SentencePiece BPE 模型。
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
        self.sp.load(model_path)

    def encode(self, text):
        return self.sp.encode_as_ids(text)

    def decode(self, ids):
        return self.sp.decode_ids(ids)

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()
