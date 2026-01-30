"""文本生成器

提供文本生成和采样功能。
"""

import torch
import torch.nn.functional as F
from typing import Optional

from ..models import MiniLLM


def generate(model: MiniLLM,
            idx: torch.Tensor,
            max_new_tokens: int,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None) -> torch.Tensor:
    """给定上下文生成新 token

    Args:
        model: 模型实例
        idx: 输入 token 索引,形状 (batch, seq_len)
        max_new_tokens: 生成的最大 token 数
        temperature: 采样温度,越高越随机
        top_k: Top-k 采样,保留概率最高的 k 个 token
        top_p: Top-p (nucleus) 采样,保留累积概率达到 p 的 token

    Returns:
        生成的完整序列,形状 (batch, seq_len + max_new_tokens)
    """
    for _ in range(max_new_tokens):
        # 如果上下文太长,进行裁剪
        idx_cond = idx[:, -model.cfg.max_seq_len:]

        # 前向传播
        logits, _ = model(idx_cond)

        # 关注最后一个时间步
        logits = logits[:, -1, :] / temperature

        # 可选: Top-k 采样
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # 可选: Top-p (Nucleus) 采样
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1
            )

            # 移除累积概率超过阈值的 token
            sorted_indices_to_remove = cumulative_probs > top_p
            # 向右移动索引,以保留第一个超过阈值的 token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float('Inf')

        # 应用 softmax 获取概率
        probs = F.softmax(logits, dim=-1)

        # 从分布中采样
        idx_next = torch.multinomial(probs, num_samples=1)

        # 将采样到的索引拼接到序列中
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


class Generator:
    """文本生成器类

    封装文本生成的完整流程。

    Args:
        model: 模型实例
        tokenizer: 分词器
        device: 计算设备

    Attributes:
        model: 模型
        tokenizer: 分词器
        device: 设备
    """

    def __init__(self, model: MiniLLM, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def generate_text(self,
                     prompt: str,
                     max_new_tokens: int = 100,
                     temperature: float = 0.8,
                     top_k: Optional[int] = 50,
                     top_p: Optional[float] = None) -> str:
        """生成文本

        Args:
            prompt: 提示词
            max_new_tokens: 生成的最大 token 数
            temperature: 采样温度
            top_k: Top-k 采样参数
            top_p: Top-p 采样参数

        Returns:
            生成的文本
        """
        # 编码提示词
        start_ids = self.tokenizer.encode(prompt)
        x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]

        # 生成
        with torch.no_grad():
            y = generate(
                self.model,
                x,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

        # 解码
        generated_text = self.tokenizer.decode(y[0].tolist())
        return generated_text

    def interactive_generate(self):
        """交互式生成

        进入交互式生成模式,用户可以输入提示词并获得生成结果。
        """
        print("=== 交互式文本生成 ===")
        print("输入提示词,模型将生成文本。输入 'quit' 退出。\n")

        while True:
            try:
                prompt = input("提示词: ")
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("退出交互模式。")
                    break

                if not prompt.strip():
                    continue

                print("\n生成中...")
                generated = self.generate_text(prompt)
                print(f"\n生成结果:\n{generated}\n")
                print("-" * 50 + "\n")

            except KeyboardInterrupt:
                print("\n\n退出交互模式。")
                break
            except Exception as e:
                print(f"错误: {e}\n")
