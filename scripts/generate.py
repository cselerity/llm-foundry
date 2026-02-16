"""命令行文本生成脚本

使用训练好的模型生成文本。

用法:
    # 基础生成
    python scripts/generate.py --model model.pt --prompt "人工智能"
    
    # 调整采样参数
    python scripts/generate.py --model model.pt --prompt "深度学习" \
        --temperature 0.8 --top-k 50 --max-tokens 100
    
    # 交互模式
    python scripts/generate.py --model model.pt --interactive
"""

import argparse
import torch

from llm_foundry import MiniLLM, Tokenizer
from llm_foundry.config import ModelConfig
from llm_foundry.inference import Generator
from llm_foundry.utils import get_device


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用 LLM 生成文本')
    
    # 模型
    parser.add_argument('--model', type=str, default='model.pt',
                       help='模型权重文件路径')
    parser.add_argument('--tokenizer', type=str, default='tokenizer.model',
                       help='分词器模型路径')
    
    # 生成模式
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--prompt', type=str, help='生成提示词')
    mode_group.add_argument('--interactive', action='store_true',
                           help='交互模式')
    
    # 生成参数
    parser.add_argument('--max-tokens', type=int, default=100,
                       help='最大生成token数')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='温度参数 (0.1-2.0)')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Top-k 采样')
    parser.add_argument('--top-p', type=float, default=None,
                       help='Top-p (nucleus) 采样')
    
    # 设备
    parser.add_argument('--device', type=str, choices=['cuda', 'mps', 'cpu', 'auto'],
                       default='auto', help='计算设备')
    
    return parser.parse_args()


def load_model(model_path, tokenizer_path, device):
    """加载模型和分词器"""
    print("加载模型...")
    
    # 加载分词器
    tokenizer = Tokenizer(tokenizer_path)
    print(f"✅ 分词器加载完成: {tokenizer_path}")
    
    # 创建模型配置（需要与训练时一致）
    # 这里使用默认配置，实际应该保存配置
    cfg = ModelConfig()
    model = MiniLLM(cfg).to(device)
    
    # 加载权重
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"✅ 模型加载完成: {model_path}")
    except FileNotFoundError:
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先训练模型或指定正确的模型路径")
        return None, None
    
    return model, tokenizer


def generate_text(generator, prompt, max_tokens, temperature, top_k, top_p):
    """生成文本"""
    print(f"\n提示词: {prompt}")
    print("-" * 70)
    
    generated = generator.generate_text(
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    
    print(generated)
    print()


def interactive_mode(generator, max_tokens, temperature, top_k, top_p):
    """交互模式"""
    print("\n" + "=" * 70)
    print("交互式生成模式")
    print("=" * 70)
    print("输入提示词生成文本，输入 'quit' 或 'exit' 退出")
    print(f"当前参数: temperature={temperature}, top_k={top_k}, "
          f"top_p={top_p}, max_tokens={max_tokens}")
    print("=" * 70)
    print()
    
    while True:
        try:
            prompt = input("提示词 >>> ").strip()
            
            if not prompt:
                continue
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("退出交互模式")
                break
            
            # 检查是否是参数调整命令
            if prompt.startswith('/'):
                handle_command(prompt)
                continue
            
            # 生成文本
            print()
            generated = generator.generate_text(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            print(generated)
            print()
            
        except KeyboardInterrupt:
            print("\n\n退出交互模式")
            break
        except Exception as e:
            print(f"错误: {e}")


def handle_command(command):
    """处理交互命令"""
    # 这里可以添加参数调整等命令
    print(f"未知命令: {command}")


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 70)
    print("LLM Foundry - 文本生成")
    print("=" * 70)
    print()
    
    # 1. 设置设备
    if args.device == 'auto':
        device = get_device()
    else:
        device = args.device
    
    print(f"使用设备: {device}")
    print()
    
    # 2. 加载模型
    model, tokenizer = load_model(args.model, args.tokenizer, device)
    if model is None:
        return 1
    
    # 3. 创建生成器
    generator = Generator(model, tokenizer, device)
    print()
    
    # 4. 生成文本
    if args.interactive:
        # 交互模式
        interactive_mode(
            generator,
            args.max_tokens,
            args.temperature,
            args.top_k,
            args.top_p
        )
    elif args.prompt:
        # 单次生成
        generate_text(
            generator,
            args.prompt,
            args.max_tokens,
            args.temperature,
            args.top_k,
            args.top_p
        )
    else:
        # 默认示例
        print("运行示例生成...")
        print()
        
        examples = [
            "人工智能",
            "深度学习",
            "满纸荒唐言，",
        ]
        
        for prompt in examples:
            generate_text(
                generator,
                prompt,
                args.max_tokens,
                args.temperature,
                args.top_k,
                args.top_p
            )
    
    print("=" * 70)
    print("提示:")
    print("  - 调整 temperature 控制随机性 (0.1-2.0)")
    print("  - 使用 top-k 限制候选词数量")
    print("  - 使用 top-p 进行核采样")
    print("  - 使用 --interactive 进入交互模式")
    print("=" * 70)
    print()
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
