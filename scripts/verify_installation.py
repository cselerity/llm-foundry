"""安装验证脚本

快速验证 LLM Foundry 的安装和环境配置。

运行:
    python scripts/verify_installation.py
"""

import sys
import importlib.util


def check_python_version():
    """检查 Python 版本"""
    print("=" * 60)
    print("1. 检查 Python 版本")
    print("=" * 60)
    
    version = sys.version_info
    print(f"Python 版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 版本过低，需要 >= 3.8")
        return False
    else:
        print("✅ Python 版本符合要求\n")
        return True


def check_dependencies():
    """检查依赖包"""
    print("=" * 60)
    print("2. 检查依赖包")
    print("=" * 60)
    
    dependencies = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'sentencepiece': 'SentencePiece',
    }
    
    all_ok = True
    for module, name in dependencies.items():
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {name:15s} {version}")
        except ImportError:
            print(f"❌ {name:15s} 未安装")
            all_ok = False
    
    print()
    return all_ok


def check_gpu():
    """检查 GPU 可用性"""
    print("=" * 60)
    print("3. 检查 GPU")
    print("=" * 60)
    
    try:
        import torch
        
        # 检查 CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"   显存: {props.total_memory / 1e9:.1f} GB")
            print(f"   计算能力: {props.major}.{props.minor}")
        else:
            print("⚠️  CUDA 不可用")
        
        # 检查 MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"✅ MPS (Apple Silicon) 可用")
        else:
            print("⚠️  MPS 不可用")
        
        # 推荐设备
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        
        print(f"\n推荐使用设备: {device}")
        print()
        return True
        
    except ImportError:
        print("❌ PyTorch 未安装，无法检查 GPU")
        print()
        return False


def check_llm_foundry():
    """检查 LLM Foundry 安装"""
    print("=" * 60)
    print("4. 检查 LLM Foundry")
    print("=" * 60)
    
    try:
        import llm_foundry
        print(f"✅ LLM Foundry 已安装")
        print(f"   版本: {llm_foundry.__version__}")
        
        # 检查主要模块
        modules = [
            'ModelConfig',
            'TrainConfig',
            'MiniLLM',
            'Tokenizer',
            'DataLoader',
            'get_device',
        ]
        
        print("\n检查模块导入:")
        all_ok = True
        for module in modules:
            try:
                getattr(llm_foundry, module)
                print(f"   ✅ {module}")
            except AttributeError:
                print(f"   ❌ {module} 不可用")
                all_ok = False
        
        print()
        return all_ok
        
    except ImportError:
        print("❌ LLM Foundry 未安装")
        print("\n请运行: pip install -e .")
        print()
        return False


def run_quick_test():
    """运行快速测试"""
    print("=" * 60)
    print("5. 快速功能测试")
    print("=" * 60)
    
    try:
        import torch
        from llm_foundry import ModelConfig, MiniLLM, get_device
        
        # 测试设备检测
        device = get_device()
        print(f"✅ 设备检测: {device}")
        
        # 测试模型创建
        cfg = ModelConfig(dim=128, n_layers=2, n_heads=4, n_kv_heads=2)
        model = MiniLLM(cfg)
        print(f"✅ 模型创建: {model.get_num_params()/1e6:.2f}M 参数")
        
        # 测试前向传播
        tokens = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, loss = model(tokens, tokens)
        print(f"✅ 前向传播: logits shape {logits.shape}")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print()
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("LLM Foundry 安装验证")
    print("=" * 60)
    print()
    
    results = []
    
    # 运行所有检查
    results.append(("Python 版本", check_python_version()))
    results.append(("依赖包", check_dependencies()))
    results.append(("GPU", check_gpu()))
    results.append(("LLM Foundry", check_llm_foundry()))
    results.append(("功能测试", run_quick_test()))
    
    # 总结
    print("=" * 60)
    print("验证总结")
    print("=" * 60)
    
    for name, ok in results:
        status = "✅ 通过" if ok else "❌ 失败"
        print(f"{name:15s} {status}")
    
    all_passed = all(ok for _, ok in results)
    
    print()
    if all_passed:
        print("🎉 所有检查通过！")
        print("\n下一步:")
        print("  1. cd tutorials")
        print("  2. python train.py")
        print("  3. python generate.py")
    else:
        print("⚠️  部分检查未通过，请根据上述信息修复问题。")
        print("\n常见问题:")
        print("  - 依赖未安装: pip install -e .")
        print("  - LLM Foundry 未安装: pip install -e .")
        print("  - GPU 不可用: 检查 CUDA/驱动安装")
    
    print()
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
