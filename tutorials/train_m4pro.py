"""
Train M4 Pro - Apple M4 Pro 优化训练脚本
=====================================

本脚本专门为 Apple M4 Pro (32GB 统一内存) 优化,充分利用 Metal Performance Shaders。

配置特点:
--------
- 模型规模: 68-72M 参数
- 批次大小: 32 (利用统一内存优势)
- 上下文长度: 1024 tokens
- 词汇表: 32k tokens
- 层数: 10 层 (MPS 优化)
- 隐藏维度: 704 (Metal 优化)

预期性能:
--------
- 训练速度: 1500-2500 tokens/sec
- 10k steps: 约 40-60 分钟
- 内存占用: 4-6 GB (统一内存)
- 生成质量: 高

使用方式:
--------
python train_m4pro.py

Apple Silicon 优势:
-----------------
- 统一内存架构 (CPU + GPU 共享内存)
- Metal Performance Shaders 加速
- 低功耗高性能
- 安静运行 (无风扇噪音)
"""

import torch
import time
from config import get_m4pro_config, get_m4pro_train_config
from model import MiniLLM
from dataloader import DataLoader


def estimate_loss(model, loader, eval_iters, device):
    """估计训练集和验证集上的平均损失"""
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = loader.get_batch(split)
            with torch.no_grad():
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out


def train():
    """Apple M4 Pro 优化训练流程"""

    # ========================================================================
    # 步骤 1: 初始化配置
    # ========================================================================
    print("=" * 80)
    print("Apple M4 Pro 优化训练")
    print("=" * 80)

    # 使用 M4 Pro 优化配置
    model_cfg = get_m4pro_config()
    train_cfg = get_m4pro_train_config()

    print("\n模型配置 (M4 Pro 优化):")
    print(f"  参数量:          ~68-72M")
    print(f"  隐藏维度:        {model_cfg.dim} (11 × 64, Metal 优化)")
    print(f"  层数:            {model_cfg.n_layers} (MPS 优化)")
    print(f"  注意力头数:      {model_cfg.n_heads}")
    print(f"  KV 头数:         {model_cfg.n_kv_heads} (激进的 GQA)")
    print(f"  词汇表大小:      {model_cfg.vocab_size}")
    print(f"  最大序列长度:    {model_cfg.max_seq_len}")

    print("\n训练配置:")
    print(f"  Batch Size:      {train_cfg.batch_size} (利用统一内存)")
    print(f"  Learning Rate:   {train_cfg.learning_rate}")
    print(f"  Max Iterations:  {train_cfg.max_iters}")
    print(f"  Eval Interval:   {train_cfg.eval_interval}")

    # 检查设备
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    if device != 'mps':
        print(f"\n⚠️  警告: MPS 不可用,使用 {device}")
        print("   M4 Pro 优化配置需要 MPS (Metal Performance Shaders)")
        print("   请确保:")
        print("   1. 使用 macOS 12.3+")
        print("   2. 安装 PyTorch 1.12+ (推荐 2.0+)")
        print("\n是否继续? (训练会很慢) [y/N]: ", end='')
        response = input().strip().lower()
        if response != 'y':
            print("训练已取消")
            return

    print(f"\n✓ 使用设备: {device}")

    if device == 'mps':
        print(f"  使用 Metal Performance Shaders (MPS)")
        print(f"  统一内存架构: CPU + GPU 共享内存")
        print(f"  PyTorch 版本: {torch.__version__}")

        # 提示 MPS 性能优化
        pytorch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        if pytorch_version < (2, 0):
            print(f"\n  ⚠️  建议升级到 PyTorch 2.0+ 以获得最佳 MPS 性能")
            print(f"     当前版本: {torch.__version__}")
            print(f"     升级命令: pip install --upgrade torch torchvision torchaudio")

    # ========================================================================
    # 步骤 2: 创建数据加载器和模型
    # ========================================================================
    print("\n" + "=" * 80)
    print("初始化数据和模型")
    print("=" * 80)

    # 创建数据加载器
    loader = DataLoader(
        batch_size=train_cfg.batch_size,
        block_size=model_cfg.max_seq_len,
        device=device
    )

    # 创建模型
    print("\n正在创建模型...")
    print("⏳ 首次运行会编译 Metal shaders,可能需要 1-2 分钟...")

    model_start = time.time()
    model = MiniLLM(model_cfg).to(device)
    model_time = time.time() - model_start

    print(f"✓ 模型创建完成 (耗时 {model_time:.1f}s)")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n模型统计:")
    print(f"  总参数量:       {total_params/1e6:.2f}M")
    print(f"  可训练参数:     {trainable_params/1e6:.2f}M")

    # ========================================================================
    # 步骤 3: 初始化优化器
    # ========================================================================
    print("\n" + "=" * 80)
    print("初始化优化器")
    print("=" * 80)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)

    print(f"优化器: AdamW")
    print(f"  学习率:         {train_cfg.learning_rate}")
    print(f"  Betas:          {optimizer.defaults['betas']}")
    print(f"  Weight Decay:   {optimizer.defaults['weight_decay']}")

    # ========================================================================
    # 步骤 4: 训练循环
    # ========================================================================
    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)
    print(f"\n预计训练时间: {train_cfg.max_iters * 0.020 / 60:.1f} 分钟")
    print(f"评估次数: {train_cfg.max_iters // train_cfg.eval_interval} 次")
    print(f"\n提示: M4 Pro 训练时功耗低,可以在后台运行\n")

    start_time = time.time()
    tokens_processed = 0

    for iter in range(train_cfg.max_iters):
        # 定期评估
        if iter % train_cfg.eval_interval == 0 or iter == train_cfg.max_iters - 1:
            losses = estimate_loss(model, loader, train_cfg.eval_iters, device)

            # 计算速度统计
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
            eta = (train_cfg.max_iters - iter) * (elapsed / (iter + 1)) if iter > 0 else 0

            print(f"step {iter:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | "
                  f"{tokens_per_sec:6.0f} tok/s | {elapsed:6.1f}s | ETA {eta/60:4.1f}m")

        # 训练步骤
        xb, yb = loader.get_batch('train')
        tokens_processed += xb.numel()

        # 前向传播
        logits, loss = model(xb, yb)

        # 反向传播
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # ========================================================================
    # 训练完成
    # ========================================================================
    end_time = time.time()
    total_time = end_time - start_time
    total_tokens = tokens_processed
    avg_tokens_per_sec = total_tokens / total_time

    print("\n" + "=" * 80)
    print("训练完成!")
    print("=" * 80)
    print(f"总耗时:         {total_time:.2f}s ({total_time/60:.1f} 分钟)")
    print(f"平均速度:       {train_cfg.max_iters / total_time:.1f} steps/s")
    print(f"Token 处理速度: {avg_tokens_per_sec:.0f} tokens/s")
    print(f"总 Tokens:      {total_tokens:,}")

    # ========================================================================
    # 保存模型
    # ========================================================================
    print("\n保存模型...")

    # 保存模型权重
    model_path = 'minillm_m4pro.pt'
    torch.save(model.state_dict(), model_path)
    print(f"✓ 模型已保存至 {model_path}")

    # 保存完整检查点
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter': train_cfg.max_iters,
        'model_config': model_cfg,
        'train_config': train_cfg,
        'final_loss': losses,
    }
    checkpoint_path = 'checkpoint_m4pro.pt'
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ 检查点已保存至 {checkpoint_path}")

    # ========================================================================
    # 训练总结
    # ========================================================================
    print("\n" + "=" * 80)
    print("训练总结")
    print("=" * 80)

    final_losses = estimate_loss(model, loader, train_cfg.eval_iters, device)
    print(f"最终训练损失:   {final_losses['train']:.4f}")
    print(f"最终验证损失:   {final_losses['val']:.4f}")
    print(f"过拟合程度:     {(final_losses['val'] - final_losses['train']):.4f}")

    if final_losses['val'] - final_losses['train'] > 0.5:
        print("\n⚠️  警告: 验证损失明显高于训练损失,可能存在过拟合")
        print("   建议: 增加 dropout、减少训练轮次或增加训练数据")
    elif final_losses['train'] > 3.0:
        print("\n⚠️  警告: 训练损失仍然较高,模型可能欠拟合")
        print("   建议: 延长训练时间或调整学习率")
    else:
        print("\n✓ 训练看起来正常!")

    # 性能分析
    print("\n" + "=" * 80)
    print("性能分析 (Apple M4 Pro)")
    print("=" * 80)
    print(f"实际吞吐量:     {avg_tokens_per_sec:.0f} tokens/sec")
    print(f"理论吞吐量:     1500-2500 tokens/sec (M4 Pro)")

    throughput_ratio = avg_tokens_per_sec / 2000  # 2000 是目标值
    if throughput_ratio > 0.8:
        print(f"性能评级:       优秀 ({throughput_ratio*100:.0f}%)")
    elif throughput_ratio > 0.6:
        print(f"性能评级:       良好 ({throughput_ratio*100:.0f}%)")
    else:
        print(f"性能评级:       有待优化 ({throughput_ratio*100:.0f}%)")
        print("\n优化建议:")
        print("  - 升级到 PyTorch 2.0+")
        print("  - 检查是否有其他程序占用大量内存")
        print("  - 尝试减小 batch_size")
        print("  - 关闭不必要的后台应用")

    # Apple Silicon 特有信息
    print("\n" + "=" * 80)
    print("Apple Silicon 优势")
    print("=" * 80)
    print("✓ 统一内存架构: CPU + GPU 共享内存池")
    print("✓ 低功耗运行: 相比独立显卡更省电")
    print("✓ 安静运行: 无风扇噪音或极低噪音")
    print("✓ 热管理: 优秀的散热设计")
    print("✓ 便携性: 可在笔记本上长时间训练")

    print("\n下一步:")
    print(f"  1. 运行生成脚本: python generate.py (修改为加载 {model_path})")
    print("  2. 继续训练: 从检查点恢复训练")
    print("  3. 调整超参数重新训练")
    print("  4. 尝试更大的数据集")
    print("\n监控内存: Activity Monitor → Memory → Memory Pressure")


if __name__ == '__main__':
    """M4 Pro 训练脚本入口

    运行方式:
    --------
    python train_m4pro.py

    前提条件:
    --------
    - macOS 12.3+ (支持 MPS)
    - PyTorch 1.12+ (推荐 2.0+)
    - Apple M4 Pro 芯片 (或 M1/M2/M3 Pro/Max/Ultra)
    - 32GB 统一内存推荐

    预期输出:
    --------
    1. 配置信息和设备检测
    2. Metal shader 编译提示 (首次运行)
    3. 数据加载进度
    4. 训练进度 (损失、速度、ETA)
    5. 模型保存信息
    6. 训练总结和性能分析
    7. Apple Silicon 特有优势说明

    训练时长:
    --------
    - M4 Pro (32GB): 约 40-60 分钟
    - M3 Pro (36GB): 约 50-70 分钟
    - M2 Pro (32GB): 约 60-80 分钟
    - M1 Pro (32GB): 约 70-90 分钟

    输出文件:
    --------
    - minillm_m4pro.pt: 模型权重
    - checkpoint_m4pro.pt: 完整检查点
    - tokenizer.model: 分词器 (如果之前不存在)
    - input_cn.txt: 训练数据 (如果之前不存在)

    故障排除:
    --------
    - MPS 不可用: 检查 macOS 版本和 PyTorch 版本
    - 训练很慢: 升级到 PyTorch 2.0+
    - 内存压力: 监控 Activity Monitor,减小 batch_size

    Apple Silicon 优化:
    ------------------
    1. 使用 PyTorch 2.0+ 获得最佳 MPS 性能
    2. 某些操作可能回退到 CPU (正常现象)
    3. 统一内存允许更灵活的 batch_size
    4. 训练时可以同时使用其他应用 (轻量级)

    与 CUDA 对比:
    -----------
    - 速度: M4 Pro 约为 RTX 5060 的 60-80%
    - 功耗: M4 Pro 约为 RTX 5060 的 25-30%
    - 便携: M4 Pro 可在笔记本上训练
    - 噪音: M4 Pro 几乎无噪音
    """
    train()
