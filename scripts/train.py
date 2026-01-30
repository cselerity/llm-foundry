#!/usr/bin/env python3
"""训练脚本

使用方式:
    python scripts/train.py
    python scripts/train.py --config configs/medium.yaml
    python scripts/train.py --dim 512 --n_layers 8
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch

from llm_foundry import ModelConfig, TrainConfig, MiniLLM, DataLoader
from llm_foundry.training import Trainer
from llm_foundry.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description='训练 LLM 模型')

    # 配置文件
    parser.add_argument('--config', type=str, default=None,
                       help='YAML 配置文件路径')

    # 模型配置
    parser.add_argument('--dim', type=int, default=256,
                       help='模型维度')
    parser.add_argument('--n_layers', type=int, default=4,
                       help='层数')
    parser.add_argument('--n_heads', type=int, default=8,
                       help='注意力头数')
    parser.add_argument('--n_kv_heads', type=int, default=4,
                       help='KV 头数')
    parser.add_argument('--vocab_size', type=int, default=8192,
                       help='词表大小')
    parser.add_argument('--max_seq_len', type=int, default=256,
                       help='最大序列长度')

    # 训练配置
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='学习率')
    parser.add_argument('--max_iters', type=int, default=1000,
                       help='最大迭代次数')
    parser.add_argument('--eval_interval', type=int, default=50,
                       help='评估间隔')

    # 数据配置
    parser.add_argument('--data_file', type=str, default='input_cn.txt',
                       help='训练数据文件')

    # 输出配置
    parser.add_argument('--output', type=str, default='minillm.pt',
                       help='模型保存路径')

    # 设备配置
    parser.add_argument('--device', type=str, default='auto',
                       help='计算设备 (cuda/mps/cpu/auto)')

    return parser.parse_args()


def main():
    args = parse_args()

    # 如果提供了配置文件,加载它
    if args.config:
        print(f"加载配置文件: {args.config}")
        # TODO: 实现 YAML 配置加载
        print("注意: YAML 配置加载尚未实现,使用命令行参数")

    # 设备检测
    if args.device == 'auto':
        device = get_device()
    else:
        device = args.device
    print(f"使用设备: {device}")

    # 创建配置
    model_cfg = ModelConfig(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len
    )

    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval
    )

    print(f"\n模型配置:")
    print(f"  维度: {model_cfg.dim}")
    print(f"  层数: {model_cfg.n_layers}")
    print(f"  注意力头数: {model_cfg.n_heads}")
    print(f"  KV 头数: {model_cfg.n_kv_heads}")
    print(f"  词表大小: {model_cfg.vocab_size}")
    print(f"  序列长度: {model_cfg.max_seq_len}")

    print(f"\n训练配置:")
    print(f"  批量大小: {train_cfg.batch_size}")
    print(f"  学习率: {train_cfg.learning_rate}")
    print(f"  最大迭代: {train_cfg.max_iters}")

    # 加载数据
    print(f"\n加载数据: {args.data_file}")
    loader = DataLoader(
        file_path=args.data_file,
        batch_size=train_cfg.batch_size,
        block_size=model_cfg.max_seq_len,
        device=device
    )

    # 创建模型
    print("\n创建模型...")
    model = MiniLLM(model_cfg).to(device)
    print(f"模型参数量: {model.get_num_params()/1e6:.2f}M")

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_config=train_cfg,
        data_loader=loader,
        device=device
    )

    # 训练
    print()
    stats = trainer.train()

    # 保存模型
    print(f"\n保存模型到: {args.output}")
    torch.save(model.state_dict(), args.output)

    print("\n训练完成! 🎉")
    print(f"  最终训练损失: {stats['train_losses'][-1]:.4f}")
    print(f"  最终验证损失: {stats['val_losses'][-1]:.4f}")
    print(f"  总耗时: {stats['elapsed_time']:.2f}s")


if __name__ == '__main__':
    main()
