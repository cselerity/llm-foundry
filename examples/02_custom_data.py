"""02_custom_data.py - 自定义数据集示例

演示如何使用自己的文本数据训练模型。

运行:
    python examples/02_custom_data.py
"""

from llm_foundry import ModelConfig, TrainConfig, MiniLLM, DataLoader
from llm_foundry.training import Trainer
from llm_foundry.utils import get_device
import torch
import os


def create_sample_data(filepath='custom_data.txt'):
    """创建示例数据文件"""
    print(f"创建示例数据文件: {filepath}")

    sample_text = """
人工智能是计算机科学的一个分支。它企图了解智能的实质,
并生产出一种新的能以人类智能相似的方式做出反应的智能机器。

机器学习是人工智能的核心,是使计算机具有智能的根本途径,
其应用遍及人工智能的各个领域。

深度学习是机器学习的一个分支,它基于人工神经网络的研究,
特别是利用多层神经网络来进行学习和表征。

近年来,深度学习在计算机视觉、自然语言处理、语音识别等领域
取得了突破性进展,推动了人工智能技术的快速发展。

大语言模型是深度学习在自然语言处理领域的重要应用,
通过在海量文本数据上进行预训练,模型能够理解和生成人类语言。

Transformer 架构的提出,为大语言模型的发展奠定了基础。
注意力机制使得模型能够更好地捕捉文本中的长程依赖关系。

现在,让我们使用 LLM Foundry 来训练一个简单的语言模型!
"""

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(sample_text * 20)  # 重复以增加数据量

    print(f"✅ 示例数据创建完成\n")


def main():
    print("=== 自定义数据集训练示例 ===\n")

    # 1. 创建自定义数据
    data_file = 'custom_data.txt'
    if not os.path.exists(data_file):
        create_sample_data(data_file)

    # 2. 设置
    device = get_device()
    print(f"使用设备: {device}\n")

    # 3. 配置 (使用较小的词表,适合小数据集)
    model_cfg = ModelConfig(
        dim=128,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=2048,  # 较小的词表
        max_seq_len=128
    )

    train_cfg = TrainConfig(
        batch_size=16,
        learning_rate=3e-4,
        max_iters=200,
        eval_interval=50
    )

    print(f"模型配置: dim={model_cfg.dim}, layers={model_cfg.n_layers}, "
          f"vocab={model_cfg.vocab_size}\n")

    # 4. 加载自定义数据
    print(f"加载自定义数据: {data_file}")
    loader = DataLoader(
        file_path=data_file,
        batch_size=train_cfg.batch_size,
        block_size=model_cfg.max_seq_len,
        device=device
    )
    print(f"数据加载完成: {len(loader)} tokens\n")

    # 5. 创建并训练模型
    print("创建模型...")
    model = MiniLLM(model_cfg).to(device)
    print(f"模型参数量: {model.get_num_params()/1e6:.2f}M\n")

    trainer = Trainer(
        model=model,
        train_config=train_cfg,
        data_loader=loader,
        device=device
    )

    print("开始训练...\n")
    stats = trainer.train()

    # 6. 保存模型
    torch.save(model.state_dict(), 'custom_model.pt')

    # 7. 测试生成
    print("\n=== 测试生成 ===")
    from llm_foundry.inference import Generator

    generator = Generator(model, loader.tokenizer, device)
    prompts = ["人工智能", "深度学习", "大语言模型"]

    for prompt in prompts:
        print(f"\n提示词: {prompt}")
        generated = generator.generate_text(
            prompt=prompt,
            max_new_tokens=50,
            temperature=0.8,
            top_k=20
        )
        print(f"生成: {generated[:100]}...")

    print("\n✅ 自定义数据训练完成!")


if __name__ == '__main__':
    main()
