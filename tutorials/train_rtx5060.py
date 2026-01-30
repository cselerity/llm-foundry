"""
Train RTX 5060 - RTX 5060 优化训练脚本
=====================================

本脚本专门为 RTX 5060 (8GB 显存) 优化,提供最佳的性能和显存平衡。

配置特点:
--------
- 模型规模: 70-75M 参数
- 批次大小: 24 (优化的显存占用)
- 上下文长度: 1024 tokens
- 词汇表: 32k tokens
- GQA 优化: 节省 50% KV cache

预期性能:
--------
- 训练速度: 2500-3500 tokens/sec
- 10k steps: 约 30-40 分钟
- 显存占用: 3-4 GB
- 生成质量: 高

使用方式:
--------
python train_rtx5060.py

或者在代码中自定义参数。
"""

import torch
import time
from config import get_rtx5060_config, get_rtx5060_train_config
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
    """RTX 5060 优化训练流程"""

    # ========================================================================
    # 步骤 1: 初始化配置
    # ========================================================================
    print("=" * 80)
    print("RTX 5060 优化训练")
    print("=" * 80)

    # 使用 RTX 5060 优化配置
    model_cfg = get_rtx5060_config()
    train_cfg = get_rtx5060_train_config()

    print("\n模型配置:")
    print(f"  参数量:          ~70-75M")
    print(f"  隐藏维度:        {model_cfg.dim}")
    print(f"  层数:            {model_cfg.n_layers}")
    print(f"  注意力头数:      {model_cfg.n_heads}")
    print(f"  KV 头数:         {model_cfg.n_kv_heads} (GQA 优化)")
    print(f"  词汇表大小:      {model_cfg.vocab_size}")
    print(f"  最大序列长度:    {model_cfg.max_seq_len}")

    print("\n训练配置:")
    print(f"  Batch Size:      {train_cfg.batch_size}")
    print(f"  Learning Rate:   {train_cfg.learning_rate}")
    print(f"  Max Iterations:  {train_cfg.max_iters}")
    print(f"  Eval Interval:   {train_cfg.eval_interval}")

    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    if device != 'cuda':
        print(f"\n⚠️  警告: 未检测到 CUDA,使用 {device}")
        print("   RTX 5060 优化配置需要 CUDA GPU")
        print("   如果您有 NVIDIA GPU,请安装 CUDA 版本的 PyTorch")
        print("\n是否继续? (训练可能很慢) [y/N]: ", end='')
        response = input().strip().lower()
        if response != 'y':
            print("训练已取消")
            return

    print(f"\n✓ 使用设备: {device}")

    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name}")
        print(f"  显存: {gpu_memory:.1f} GB")

        # 检查是否是推荐的 GPU
        if 'RTX' in gpu_name or 'GTX' in gpu_name:
            print(f"  ✓ 检测到 NVIDIA GPU,配置已优化")
        else:
            print(f"  ⚠️  建议使用 RTX 系列 GPU 以获得最佳性能")

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
    model = MiniLLM(model_cfg).to(device)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n模型统计:")
    print(f"  总参数量:       {total_params/1e6:.2f}M")
    print(f"  可训练参数:     {trainable_params/1e6:.2f}M")

    # 显存占用估算
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"\n显存占用 (初始化后):")
        print(f"  已分配:         {allocated:.2f} GB")
        print(f"  已预留:         {reserved:.2f} GB")

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
    print(f"\n预计训练时间: {train_cfg.max_iters * 0.012 / 60:.1f} 分钟")
    print(f"评估次数: {train_cfg.max_iters // train_cfg.eval_interval} 次\n")

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

            # 显示显存使用情况 (每 1000 步)
            if device == 'cuda' and iter % 1000 == 0:
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"         GPU: {allocated:.2f}GB allocated | {reserved:.2f}GB reserved | {peak:.2f}GB peak")

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

    if device == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"峰值显存:       {peak_memory:.2f} GB")

    # ========================================================================
    # 保存模型
    # ========================================================================
    print("\n保存模型...")

    # 保存模型权重
    model_path = 'minillm_rtx5060.pt'
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
    checkpoint_path = 'checkpoint_rtx5060.pt'
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
    print("性能分析")
    print("=" * 80)
    print(f"实际吞吐量:     {avg_tokens_per_sec:.0f} tokens/sec")
    print(f"理论吞吐量:     2500-3500 tokens/sec (RTX 5060)")

    throughput_ratio = avg_tokens_per_sec / 3000  # 3000 是目标值
    if throughput_ratio > 0.8:
        print(f"性能评级:       优秀 ({throughput_ratio*100:.0f}%)")
    elif throughput_ratio > 0.6:
        print(f"性能评级:       良好 ({throughput_ratio*100:.0f}%)")
    else:
        print(f"性能评级:       有待优化 ({throughput_ratio*100:.0f}%)")
        print("\n优化建议:")
        print("  - 确保使用 PyTorch 2.0+")
        print("  - 检查是否有其他程序占用 GPU")
        print("  - 尝试减小 batch_size 或 seq_len")

    print("\n下一步:")
    print(f"  1. 运行生成脚本: python generate.py (加载 {model_path})")
    print("  2. 继续训练: 从检查点恢复训练")
    print("  3. 调整超参数重新训练")
    print("  4. 尝试更大的数据集")


if __name__ == '__main__':
    """训练脚本入口

    运行方式:
    --------
    python train_rtx5060.py

    前提条件:
    --------
    - CUDA 版本的 PyTorch
    - RTX 5060 或同等性能 GPU (8GB+ 显存)
    - 训练数据 (自动下载)

    预期输出:
    --------
    1. 配置信息和硬件检测
    2. 数据加载进度
    3. 训练进度 (损失、速度、ETA)
    4. 显存使用情况
    5. 模型保存信息
    6. 训练总结和性能分析

    训练时长:
    --------
    - RTX 5060 (8GB): 约 30-40 分钟
    - RTX 3060 (12GB): 约 25-35 分钟
    - RTX 4060 Ti (16GB): 约 20-30 分钟

    输出文件:
    --------
    - minillm_rtx5060.pt: 模型权重
    - checkpoint_rtx5060.pt: 完整检查点
    - tokenizer.model: 分词器 (如果之前不存在)
    - input_cn.txt: 训练数据 (如果之前不存在)

    故障排除:
    --------
    - CUDA out of memory: 减小 batch_size (在 config.py 中)
    - 训练很慢: 确保使用 GPU (检查 nvidia-smi)
    - 损失为 NaN: 降低学习率或检查数据

    优化建议:
    --------
    1. 使用 PyTorch 2.0+ 获得最佳性能
    2. 启用 torch.compile() 进一步加速:
       model = torch.compile(model)
    3. 使用混合精度训练节省显存:
       from torch.cuda.amp import autocast, GradScaler
    4. 监控 GPU 利用率:
       watch -n 1 nvidia-smi
    """
    train()
