"""03_generation_sampling.py - 生成采样策略示例

演示不同的文本生成采样策略及其效果。

运行:
    python examples/03_generation_sampling.py
"""

from llm_foundry import ModelConfig, MiniLLM, Tokenizer
from llm_foundry.inference import Generator
from llm_foundry.utils import get_device
import torch


def load_model():
    """加载预训练模型"""
    device = get_device()
    print(f"使用设备: {device}\n")

    # 加载模型
    cfg = ModelConfig()
    model = MiniLLM(cfg).to(device)

    try:
        model.load_state_dict(torch.load('minillm.pt', map_location=device))
        print("✅ 模型加载成功")
    except FileNotFoundError:
        print("❌ 未找到模型文件 minillm.pt")
        print("请先运行 simple/train.py 或 scripts/train.py 训练模型")
        return None, None, None

    tokenizer = Tokenizer('tokenizer.model')
    generator = Generator(model, tokenizer, device)

    return generator, device, tokenizer


def demo_temperature():
    """演示温度参数的效果"""
    print("=" * 60)
    print("演示 1: 温度 (Temperature) 参数")
    print("=" * 60)
    print("温度控制随机性: 低温度更确定,高温度更随机\n")

    generator, device, tokenizer = load_model()
    if generator is None:
        return

    prompt = "满纸荒唐言，"
    temperatures = [0.3, 0.8, 1.5]

    for temp in temperatures:
        print(f"\n--- 温度 = {temp} ---")
        generated = generator.generate_text(
            prompt=prompt,
            max_new_tokens=50,
            temperature=temp,
            top_k=None,  # 不使用 top-k
            top_p=None   # 不使用 top-p
        )
        print(generated)


def demo_top_k():
    """演示 Top-k 采样"""
    print("\n" + "=" * 60)
    print("演示 2: Top-k 采样")
    print("=" * 60)
    print("Top-k 只从概率最高的 k 个 token 中采样\n")

    generator, device, tokenizer = load_model()
    if generator is None:
        return

    prompt = "满纸荒唐言，"
    top_k_values = [5, 20, 50]

    for k in top_k_values:
        print(f"\n--- Top-k = {k} ---")
        generated = generator.generate_text(
            prompt=prompt,
            max_new_tokens=50,
            temperature=0.8,
            top_k=k,
            top_p=None
        )
        print(generated)


def demo_top_p():
    """演示 Top-p (Nucleus) 采样"""
    print("\n" + "=" * 60)
    print("演示 3: Top-p (Nucleus) 采样")
    print("=" * 60)
    print("Top-p 从累积概率达到 p 的最小 token 集合中采样\n")

    generator, device, tokenizer = load_model()
    if generator is None:
        return

    prompt = "满纸荒唐言，"
    top_p_values = [0.5, 0.9, 0.95]

    for p in top_p_values:
        print(f"\n--- Top-p = {p} ---")
        generated = generator.generate_text(
            prompt=prompt,
            max_new_tokens=50,
            temperature=0.8,
            top_k=None,
            top_p=p
        )
        print(generated)


def demo_combined():
    """演示组合策略"""
    print("\n" + "=" * 60)
    print("演示 4: 组合策略 (Top-k + Top-p)")
    print("=" * 60)
    print("组合使用 Top-k 和 Top-p 可以获得更好的平衡\n")

    generator, device, tokenizer = load_model()
    if generator is None:
        return

    prompt = "满纸荒唐言，"
    configs = [
        {"temp": 0.7, "top_k": 40, "top_p": 0.9, "name": "平衡配置"},
        {"temp": 0.5, "top_k": 20, "top_p": 0.85, "name": "保守配置"},
        {"temp": 1.0, "top_k": 50, "top_p": 0.95, "name": "创意配置"},
    ]

    for config in configs:
        print(f"\n--- {config['name']} ---")
        print(f"temperature={config['temp']}, top_k={config['top_k']}, "
              f"top_p={config['top_p']}")
        generated = generator.generate_text(
            prompt=prompt,
            max_new_tokens=50,
            temperature=config['temp'],
            top_k=config['top_k'],
            top_p=config['top_p']
        )
        print(generated)


def main():
    print("\n=== 文本生成采样策略示例 ===\n")

    # 运行各个演示
    demo_temperature()
    demo_top_k()
    demo_top_p()
    demo_combined()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成!")
    print("=" * 60)
    print("\n建议:")
    print("- 创意写作: temperature=0.8-1.0, top_k=50, top_p=0.9")
    print("- 事实性文本: temperature=0.3-0.5, top_k=20, top_p=0.8")
    print("- 代码生成: temperature=0.2, top_k=10, top_p=0.9")


if __name__ == '__main__':
    main()
