"""训练器

提供模型训练功能。
"""

import torch
import time
from typing import Optional, Dict

from ..config import TrainConfig
from ..models import MiniLLM
from ..data import DataLoader


def estimate_loss(model: MiniLLM, loader: DataLoader,
                 eval_iters: int, device: str) -> Dict[str, float]:
    """估计训练集和验证集上的 loss

    Args:
        model: 模型
        loader: 数据加载器
        eval_iters: 评估迭代次数
        device: 设备

    Returns:
        包含 'train' 和 'val' loss 的字典
    """
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = loader.get_batch(split)
            with torch.no_grad():
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()

    model.train()
    return out


class Trainer:
    """训练器类

    封装模型训练的完整流程。

    Args:
        model: 要训练的模型
        train_config: 训练配置
        data_loader: 数据加载器
        device: 计算设备

    Attributes:
        model: 模型实例
        cfg: 训练配置
        loader: 数据加载器
        device: 设备
        optimizer: 优化器
    """

    def __init__(self,
                 model: MiniLLM,
                 train_config: TrainConfig,
                 data_loader: DataLoader,
                 device: str):
        self.model = model
        self.cfg = train_config
        self.loader = data_loader
        self.device = device

        # 创建优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.learning_rate
        )

    def train(self, max_iters: Optional[int] = None) -> Dict:
        """训练模型

        Args:
            max_iters: 最大迭代次数,如果为 None 则使用配置中的值

        Returns:
            训练统计信息字典
        """
        if max_iters is None:
            max_iters = self.cfg.max_iters

        print(f"开始训练 (共 {max_iters} 步)...")
        start_time = time.time()

        self.model.train()
        train_losses = []
        val_losses = []

        for iter in range(max_iters):
            # 评估
            if iter % self.cfg.eval_interval == 0 or iter == max_iters - 1:
                losses = estimate_loss(
                    self.model,
                    self.loader,
                    self.cfg.eval_iters,
                    self.device
                )
                train_losses.append(losses['train'])
                val_losses.append(losses['val'])

                print(f"step {iter}: "
                      f"train loss {losses['train']:.4f}, "
                      f"val loss {losses['val']:.4f}")

            # 采样 batch
            xb, yb = self.loader.get_batch('train')

            # 前向传播
            logits, loss = self.model(xb, yb)

            # 反向传播
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        end_time = time.time()
        elapsed = end_time - start_time

        print(f"训练完成,耗时 {elapsed:.2f}s")
        print(f"平均速度: {max_iters/elapsed:.2f} step/s")

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'elapsed_time': elapsed,
            'steps_per_second': max_iters / elapsed
        }

    def save_checkpoint(self, filepath: str):
        """保存检查点

        Args:
            filepath: 保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"检查点已保存至 {filepath}")

    def load_checkpoint(self, filepath: str):
        """加载检查点

        Args:
            filepath: 检查点文件路径
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"检查点已从 {filepath} 加载")
