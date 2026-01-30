#!/usr/bin/env python3
"""文本生成脚本

使用方式:
    python scripts/generate.py
    python scripts/generate.py --checkpoint model.pt --prompt "你好"
    python scripts/generate.py --interactive
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch

from llm_foundry import ModelConfig, MiniLLM, Tokenizer
from llm_foundry.inference import Generator
from llm_foundry.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description='生成文本')

    # 模型配置
    parser.add_argument('--checkpoint', type=str, default='minillm.pt',
                       help='模型检查点路径')
    parser.add_argument('--tokenizer', type=str, default='tokenizer.model',
                       help='分词器模型路径')

    # 生成配置
    parser.add_argument('--prompt', type=str, default='满纸荒唐言，',
                       help='提示词')
    parser.add_argument('--max_tokens', type=int, default=100,
                       help='生成的最大 token 数')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='采样温度')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k 采样参数')
    parser.add_argument('--top_p', type=float, default=None,
                       help='Top-p 采样参数')

    # 模式
    parser.add_argument('--interactive', action='store_true',
                       help='交互式生成模式')

    # 设备
    parser.add_argument('--device', type=str, default='auto',
                       help='计算设备 (cuda/mps/cpu/auto)')

    return parser.parse_args()


def main():
    args = parse_args()

    # 设备检测
    if args.device == 'auto':
        device = get_device()
    else:
        device = args.device
    print(f"使用设备: {device}")

    # 创建模型配置
    # TODO: 从检查点中加载配置
    cfg = ModelConfig()

    # 加载模型
    print(f"加载模型: {args.checkpoint}")
    model = MiniLLM(cfg).to(device)
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print("✅ 模型加载成功")
    except FileNotFoundError:
        print(f"❌ 未找到检查点文件: {args.checkpoint}")
        print("请先运行 train.py 训练模型")
        return

    model.eval()

    # 加载分词器
    print(f"加载分词器: {args.tokenizer}")
    tokenizer = Tokenizer(args.tokenizer)
    print(f"✅ 分词器加载成功 (词表大小: {tokenizer.vocab_size})")

    # 创建生成器
    generator = Generator(model, tokenizer, device)

    # 交互模式
    if args.interactive:
        generator.interactive_generate()
        return

    # 单次生成模式
    print(f"\n提示词: {args.prompt}")
    print(f"生成参数: temperature={args.temperature}, top_k={args.top_k}, "
          f"max_tokens={args.max_tokens}")
    print("\n正在生成...")

    generated_text = generator.generate_text(
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    print("\n" + "=" * 50)
    print("生成结果:")
    print("=" * 50)
    print(generated_text)
    print("=" * 50 + "\n")


if __name__ == '__main__':
    main()
