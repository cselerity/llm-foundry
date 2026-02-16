"""命令行训练脚本

使用配置文件或预设配置训练模型。

用法:
    # 使用预设配置
    python scripts/train.py --preset small
    
    # 使用配置文件
    python scripts/train.py --config configs/medium.yaml
    
    # 自定义参数
    python scripts/train.py --preset small --batch-size 16 --max-iters 2000
"""

import argparse
import torch
from pathlib import Path

from llm_foundry import MiniLLM, DataLoader
from llm_foundry.config import load_config, get_preset_config
from llm_foundry.training import Trainer
from llm_foundry.utils import get_device


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练 LLM 模型')
    
    # 配置来源
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument('--config', type=str, help='配置文件路径')
    config_group.add_argument('--preset', type=str, 
                             choices=['small', 'medium', 'rtx5060', 'm4pro'],
                             help='预设配置名称')
    
    # 数据
    parser.add_argument('--data', type=str, default='input_cn.txt',
                       help='训练数据文件路径')
    
    # 训练参数覆盖
    parser.add_argument('--batch-size', type=int, help='批次大小')
    parser.add_argument('--learning-rate', type=float, help='学习率')
    parser.add_argument('--max-iters', type=int, help='最大迭代次数')
    parser.add_argument('--eval-interval', type=int, help='评估间隔')
    
    # 输出
    parser.add_argument('--output', type=str, default='model.pt',
                       help='模型保存路径')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='检查点保存目录')
    
    # 设备
    parser.add_argument('--device', type=str, choices=['cuda', 'mps', 'cpu', 'auto'],
                       default='auto', help='计算设备')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 70)
    print("LLM Foundry - 模型训练")
    print("=" * 70)
    print()
    
    # 1. 加载配置
    print("1. 加载配置")
    print("-" * 70)
    
    if args.config:
        print(f"从文件加载: {args.config}")
        model_cfg, train_cfg = load_config(args.config)
    else:
        print(f"使用预设: {args.preset}")
        model_cfg, train_cfg = get_preset_config(args.preset)
    
    # 覆盖参数
    if args.batch_size:
        train_cfg.batch_size = args.batch_size
    if args.learning_rate:
        train_cfg.learning_rate = args.learning_rate
    if args.max_iters:
        train_cfg.max_iters = args.max_iters
    if args.eval_interval:
        train_cfg.eval_interval = args.eval_interval
    
    print(f"模型配置: dim={model_cfg.dim}, layers={model_cfg.n_layers}, "
          f"heads={model_cfg.n_heads}")
    print(f"训练配置: batch_size={train_cfg.batch_size}, lr={train_cfg.learning_rate}, "
          f"max_iters={train_cfg.max_iters}")
    print()
    
    # 2. 设置设备
    print("2. 设置设备")
    print("-" * 70)
    
    if args.device == 'auto':
        device = get_device()
    else:
        device = args.device
    
    print(f"使用设备: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"显存: {props.total_memory / 1e9:.1f} GB")
    
    print()
    
    # 3. 加载数据
    print("3. 加载数据")
    print("-" * 70)
    
    print(f"数据文件: {args.data}")
    loader = DataLoader(
        file_path=args.data,
        batch_size=train_cfg.batch_size,
        block_size=model_cfg.max_seq_len,
        device=device
    )
    print(f"数据加载完成: {len(loader)} tokens")
    print()
    
    # 4. 创建模型
    print("4. 创建模型")
    print("-" * 70)
    
    model = MiniLLM(model_cfg).to(device)
    num_params = model.get_num_params()
    print(f"模型参数量: {num_params/1e6:.2f}M")
    print(f"非嵌入参数: {model.get_num_params(non_embedding=True)/1e6:.2f}M")
    print()
    
    # 5. 训练
    print("5. 开始训练")
    print("-" * 70)
    
    trainer = Trainer(
        model=model,
        train_config=train_cfg,
        data_loader=loader,
        device=device
    )
    
    stats = trainer.train()
    
    print()
    
    # 6. 保存模型
    print("6. 保存模型")
    print("-" * 70)
    
    # 保存模型权重
    torch.save(model.state_dict(), args.output)
    print(f"✅ 模型已保存到: {args.output}")
    
    # 保存检查点
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / 'final_checkpoint.pt'
    trainer.save_checkpoint(str(checkpoint_path))
    print(f"✅ 检查点已保存到: {checkpoint_path}")
    
    print()
    
    # 7. 训练总结
    print("7. 训练总结")
    print("-" * 70)
    
    print(f"最终训练损失: {stats['train_losses'][-1]:.4f}")
    print(f"最终验证损失: {stats['val_losses'][-1]:.4f}")
    print(f"总耗时: {stats['elapsed_time']:.2f}s ({stats['elapsed_time']/60:.1f} 分钟)")
    print(f"训练速度: {stats['steps_per_second']:.2f} steps/s")
    
    # 过拟合检测
    overfitting = stats['val_losses'][-1] - stats['train_losses'][-1]
    if overfitting > 0.5:
        print(f"\n⚠️  警告: 可能存在过拟合 (差距: {overfitting:.4f})")
        print("   建议: 增加正则化、减少模型大小或增加训练数据")
    elif stats['train_losses'][-1] > 3.0:
        print(f"\n⚠️  警告: 训练损失较高，可能欠拟合")
        print("   建议: 增加模型大小、延长训练时间或调整学习率")
    else:
        print(f"\n✅ 训练完成，模型表现正常")
    
    print()
    print("=" * 70)
    print("下一步:")
    print("  1. 使用 scripts/generate.py 生成文本")
    print("  2. 调整配置重新训练")
    print("  3. 在自己的数据上训练")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()
