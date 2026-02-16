"""04_config_management.py - 配置管理示例

演示如何使用统一的配置管理系统。

运行:
    python examples/04_config_management.py
"""

from llm_foundry.config import (
    ModelConfig, 
    TrainConfig,
    load_config,
    save_config,
    get_preset_config
)


def demo_direct_creation():
    """演示直接创建配置"""
    print("=" * 60)
    print("方法 1: 直接创建配置")
    print("=" * 60)
    
    model_cfg = ModelConfig(
        dim=256,
        n_layers=4,
        n_heads=8,
        n_kv_heads=4
    )
    
    train_cfg = TrainConfig(
        batch_size=32,
        learning_rate=3e-4,
        max_iters=1000
    )
    
    print(f"模型配置: dim={model_cfg.dim}, layers={model_cfg.n_layers}")
    print(f"训练配置: batch_size={train_cfg.batch_size}, lr={train_cfg.learning_rate}")
    print()


def demo_preset_config():
    """演示使用预设配置"""
    print("=" * 60)
    print("方法 2: 使用预设配置")
    print("=" * 60)
    
    presets = ['small', 'medium', 'rtx5060', 'm4pro']
    
    for preset in presets:
        model_cfg, train_cfg = get_preset_config(preset)
        params = model_cfg.dim * model_cfg.n_layers * 4  # 粗略估算
        print(f"\n{preset.upper()} 配置:")
        print(f"  模型: dim={model_cfg.dim}, layers={model_cfg.n_layers}")
        print(f"  训练: batch_size={train_cfg.batch_size}, iters={train_cfg.max_iters}")
        print(f"  预估参数: ~{params/1e6:.1f}M")
    
    print()


def demo_yaml_config():
    """演示 YAML 配置文件"""
    print("=" * 60)
    print("方法 3: YAML 配置文件")
    print("=" * 60)
    
    # 加载现有配置
    try:
        model_cfg, train_cfg = load_config('configs/small.yaml')
        print("✅ 成功加载 configs/small.yaml")
        print(f"   模型: dim={model_cfg.dim}, layers={model_cfg.n_layers}")
        print(f"   训练: batch_size={train_cfg.batch_size}")
    except FileNotFoundError:
        print("⚠️  configs/small.yaml 不存在")
    
    print()


def demo_save_config():
    """演示保存配置"""
    print("=" * 60)
    print("方法 4: 保存自定义配置")
    print("=" * 60)
    
    # 创建自定义配置
    model_cfg = ModelConfig(
        dim=384,
        n_layers=6,
        n_heads=6,
        n_kv_heads=3,
        vocab_size=8192,
        max_seq_len=384
    )
    
    train_cfg = TrainConfig(
        batch_size=24,
        learning_rate=2e-4,
        max_iters=3000,
        eval_interval=75
    )
    
    # 保存到文件
    output_path = 'my_custom_config.yaml'
    save_config(model_cfg, train_cfg, output_path)
    print(f"✅ 配置已保存到: {output_path}")
    
    # 重新加载验证
    loaded_model_cfg, loaded_train_cfg = load_config(output_path)
    print(f"✅ 验证加载: dim={loaded_model_cfg.dim}, batch_size={loaded_train_cfg.batch_size}")
    
    print()


def demo_config_comparison():
    """演示配置对比"""
    print("=" * 60)
    print("配置对比")
    print("=" * 60)
    
    configs = {
        'Small': get_preset_config('small'),
        'Medium': get_preset_config('medium'),
        'RTX 5060': get_preset_config('rtx5060'),
    }
    
    print(f"\n{'配置':<12} {'维度':<8} {'层数':<6} {'头数':<6} {'Batch':<8} {'迭代':<8}")
    print("-" * 60)
    
    for name, (model_cfg, train_cfg) in configs.items():
        print(f"{name:<12} {model_cfg.dim:<8} {model_cfg.n_layers:<6} "
              f"{model_cfg.n_heads:<6} {train_cfg.batch_size:<8} {train_cfg.max_iters:<8}")
    
    print()


def main():
    print("\n=== 配置管理示例 ===\n")
    
    # 运行各个演示
    demo_direct_creation()
    demo_preset_config()
    demo_yaml_config()
    demo_save_config()
    demo_config_comparison()
    
    print("=" * 60)
    print("✅ 配置管理演示完成!")
    print("=" * 60)
    
    print("\n使用建议:")
    print("  - 快速实验: 使用预设配置 get_preset_config('small')")
    print("  - 自定义训练: 直接创建 ModelConfig() 和 TrainConfig()")
    print("  - 配置复用: 保存到 YAML 文件，使用 load_config()")
    print("  - 团队协作: 使用 YAML 文件管理配置")
    print()


if __name__ == '__main__':
    main()
