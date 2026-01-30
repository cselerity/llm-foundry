"""01_basic_training.py - 基础训练示例

演示如何使用 LLM Foundry 进行基本的模型训练。

运行:
    python examples/01_basic_training.py
"""

from llm_foundry import ModelConfig, TrainConfig, MiniLLM, DataLoader
from llm_foundry.training import Trainer
from llm_foundry.utils import get_device
import torch


def main():
    print("=== 基础训练示例 ===\n")

    # 1. 设置设备
    device = get_device()
    print(f"使用设备: {device}\n")

    # 2. 创建模型配置
    model_cfg = ModelConfig(
        dim=256,
        n_layers=4,
        n_heads=8,
        n_kv_heads=4,
        vocab_size=8192,
        max_seq_len=256
    )
    print(f"模型配置: {model_cfg}\n")

    # 3. 创建训练配置
    train_cfg = TrainConfig(
        batch_size=32,
        learning_rate=3e-4,
        max_iters=100,  # 演示用,实际训练建议 1000+
        eval_interval=25
    )
    print(f"训练配置: {train_cfg}\n")

    # 4. 加载数据
    print("加载数据...")
    loader = DataLoader(
        file_path='input_cn.txt',
        batch_size=train_cfg.batch_size,
        block_size=model_cfg.max_seq_len,
        device=device
    )
    print(f"数据加载完成: {len(loader)} tokens\n")

    # 5. 创建模型
    print("创建模型...")
    model = MiniLLM(model_cfg).to(device)
    num_params = model.get_num_params()
    print(f"模型参数量: {num_params/1e6:.2f}M\n")

    # 6. 创建训练器
    trainer = Trainer(
        model=model,
        train_config=train_cfg,
        data_loader=loader,
        device=device
    )

    # 7. 训练模型
    print("开始训练...\n")
    stats = trainer.train()

    # 8. 保存模型
    output_path = 'example_model.pt'
    print(f"\n保存模型到: {output_path}")
    torch.save(model.state_dict(), output_path)

    # 9. 显示统计信息
    print("\n=== 训练统计 ===")
    print(f"最终训练损失: {stats['train_losses'][-1]:.4f}")
    print(f"最终验证损失: {stats['val_losses'][-1]:.4f}")
    print(f"总耗时: {stats['elapsed_time']:.2f}s")
    print(f"速度: {stats['steps_per_second']:.2f} steps/s")

    print("\n✅ 训练完成!")


if __name__ == '__main__':
    main()
