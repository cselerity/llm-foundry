"""
Train - 语言模型训练脚本
======================

本模块实现了完整的语言模型训练流程,从初始化到保存模型。

教学目标:
--------
- 理解语言模型的训练循环
- 掌握训练/验证损失评估
- 学习 PyTorch 的训练最佳实践
- 理解优化器和反向传播

核心概念:
--------
1. **训练循环 (Training Loop)**:
   重复执行前向传播、损失计算、反向传播、参数更新

2. **损失函数 (Loss Function)**:
   衡量模型预测与真实标签的差异
   语言模型使用交叉熵损失 (Cross-Entropy Loss)

3. **优化器 (Optimizer)**:
   根据梯度更新模型参数
   AdamW 是现代 Transformer 的标准选择

4. **评估 (Evaluation)**:
   定期在验证集上评估模型性能
   监控过拟合和训练进度

训练流程:
--------
    初始化配置
        ↓
    创建数据加载器
        ↓
    初始化模型和优化器
        ↓
    ┌──────────────┐
    │  训练循环    │  ← 重复 max_iters 次
    ├──────────────┤
    │ 1. 获取批次  │
    │ 2. 前向传播  │
    │ 3. 计算损失  │
    │ 4. 反向传播  │
    │ 5. 更新参数  │
    │ 6. 周期评估  │
    └──────────────┘
        ↓
    保存模型

为什么用这些超参数?
-----------------
- **AdamW**: Adam + Weight Decay,防止过拟合
- **zero_grad(set_to_none=True)**: 更高效的梯度清零
- **model.train() / model.eval()**: 切换训练/评估模式
- **torch.no_grad()**: 评估时不计算梯度,节省内存

训练技巧:
--------
- 定期保存检查点 (避免训练中断丢失进度)
- 监控训练/验证损失的差距 (检测过拟合)
- 使用学习率调度器 (更稳定的收敛)
- 梯度裁剪 (防止梯度爆炸)

与主工程的关系:
-------------
本文件是 src/llm_foundry/training/ 的教学展示版本。
主工程支持分布式训练、混合精度、检查点等高级功能。
"""

import torch
import time
from config import ModelConfig, TrainConfig
from model import MiniLLM
from dataloader import DataLoader


# ============================================================================
# 辅助函数: 损失评估
# ============================================================================

def estimate_loss(model, loader, eval_iters, device):
    """估计训练集和验证集上的平均损失

    教学要点:
    -------
    - 评估时禁用 dropout 和其他随机性
    - 使用多个批次的平均值,更准确
    - 评估不需要梯度,节省内存
    - 评估后恢复训练模式

    参数:
    ----
    model : MiniLLM
        待评估的模型
    loader : DataLoader
        数据加载器
    eval_iters : int
        评估时使用的批次数量
        更多批次 → 更准确但更慢
    device : str
        计算设备 ('cpu', 'cuda', 'mps')

    返回:
    ----
    dict
        {'train': train_loss, 'val': val_loss}

    评估流程:
    --------
    1. 切换到评估模式 (model.eval())
       - 禁用 Dropout
       - 禁用 Batch Normalization 更新
    2. 对训练集和验证集分别评估
    3. 每个 split 使用 eval_iters 个批次
    4. 计算平均损失
    5. 恢复训练模式 (model.train())

    为什么需要 model.eval()?
    -----------------------
    训练模式和评估模式的行为不同:
    - 训练模式: Dropout 随机丢弃神经元,增加正则化
    - 评估模式: Dropout 关闭,使用完整模型
    如果评估时不关闭 Dropout,结果会有随机性,不可比较。

    为什么用多个批次?
    ----------------
    单个批次的损失可能不稳定 (因为批次是随机采样的)。
    使用多个批次的平均值,可以得到更准确的估计。
    典型值: eval_iters = 50-200

    内存优化:
    --------
    使用 torch.no_grad() 禁用梯度计算:
    - 减少内存占用 (不存储中间激活值)
    - 加快评估速度 (跳过梯度计算)
    - 评估时不需要梯度 (不会更新参数)

    示例:
    ----
    >>> losses = estimate_loss(model, loader, eval_iters=100, device='cuda')
    >>> print(f"Train: {losses['train']:.4f}, Val: {losses['val']:.4f}")
    Train: 2.3456, Val: 2.4567
    """
    out = {}
    model.eval()  # 切换到评估模式

    # 对训练集和验证集分别评估
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)  # 存储每个批次的损失

        # 评估 eval_iters 个批次
        for k in range(eval_iters):
            X, Y = loader.get_batch(split)

            # 前向传播 (不计算梯度)
            with torch.no_grad():
                logits, loss = model(X, Y)

            losses[k] = loss.item()  # 记录损失值

        # 计算平均损失
        out[split] = losses.mean()

    model.train()  # 恢复训练模式
    return out


# ============================================================================
# 主训练函数
# ============================================================================

def train():
    """执行完整的训练流程

    教学要点:
    -------
    - 自动检测可用设备 (CUDA / MPS / CPU)
    - 训练循环的标准实现
    - 定期评估和进度输出
    - 模型保存和加载

    训练步骤:
    --------
    1. 初始化配置
    2. 创建数据加载器和模型
    3. 初始化优化器
    4. 训练循环
    5. 保存模型

    性能优化:
    --------
    - GPU 加速: 使用 CUDA 或 MPS
    - 高效梯度清零: zero_grad(set_to_none=True)
    - 批量处理: 并行处理多个样本

    监控训练:
    --------
    - 观察训练/验证损失的变化
    - 训练损失应该持续下降
    - 验证损失先降后升 → 过拟合
    - 两者差距过大 → 过拟合

    常见问题:
    --------
    - **损失不下降**: 学习率过高/过低,数据问题
    - **损失爆炸**: 梯度爆炸,需要梯度裁剪
    - **过拟合**: 验证损失上升,需要正则化
    - **欠拟合**: 两者都很高,模型太小或训练不足
    """

    # ========================================================================
    # 步骤 1: 初始化配置
    # ========================================================================
    print("=" * 60)
    print("初始化训练配置")
    print("=" * 60)

    cfg = TrainConfig()
    model_cfg = ModelConfig()

    print(f"训练配置:")
    print(f"  Batch Size:      {cfg.batch_size}")
    print(f"  Learning Rate:   {cfg.learning_rate}")
    print(f"  Max Iterations:  {cfg.max_iters}")
    print(f"  Eval Interval:   {cfg.eval_interval}")

    # 自动检测设备
    # 优先级: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\n使用设备: {device}")

    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif device == 'mps':
        print("  使用 Apple Silicon GPU")

    # ========================================================================
    # 步骤 2: 创建数据加载器和模型
    # ========================================================================
    print("\n" + "=" * 60)
    print("初始化数据和模型")
    print("=" * 60)

    # 创建数据加载器
    # 会自动下载数据和训练分词器 (如果需要)
    loader = DataLoader(batch_size=cfg.batch_size, block_size=model_cfg.max_seq_len, device=device)

    # 创建模型并移动到设备
    model = MiniLLM(model_cfg).to(device)

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n模型统计:")
    print(f"  总参数量:       {total_params/1e6:.2f}M")
    print(f"  可训练参数:     {trainable_params/1e6:.2f}M")
    print(f"  词汇表大小:     {model_cfg.vocab_size}")
    print(f"  模型维度:       {model_cfg.dim}")
    print(f"  层数:           {model_cfg.n_layers}")
    print(f"  注意力头数:     {model_cfg.n_heads}")

    # ========================================================================
    # 步骤 3: 初始化优化器
    # ========================================================================
    print("\n" + "=" * 60)
    print("初始化优化器")
    print("=" * 60)

    # AdamW: Adam with Weight Decay
    # - Adam: 自适应学习率,对每个参数独立调整
    # - Weight Decay: L2 正则化,防止过拟合
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    print(f"优化器: AdamW")
    print(f"  学习率:         {cfg.learning_rate}")
    print(f"  Weight Decay:   {optimizer.defaults.get('weight_decay', 0.01)}")

    # ========================================================================
    # 步骤 4: 训练循环
    # ========================================================================
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    start_time = time.time()

    for iter in range(cfg.max_iters):
        # --------------------------------------------------------------------
        # 定期评估
        # --------------------------------------------------------------------
        if iter % cfg.eval_interval == 0 or iter == cfg.max_iters - 1:
            losses = estimate_loss(model, loader, cfg.eval_iters, device)

            # 输出进度
            elapsed = time.time() - start_time
            print(f"step {iter:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | time {elapsed:.1f}s")

        # --------------------------------------------------------------------
        # 训练步骤
        # --------------------------------------------------------------------

        # 1. 获取批次数据
        xb, yb = loader.get_batch('train')

        # 2. 前向传播
        # 模型接收输入 xb 和目标 yb,返回 logits 和 loss
        logits, loss = model(xb, yb)

        # 3. 反向传播
        # 清空之前的梯度 (必须!)
        # set_to_none=True: 更高效的梯度清零方式
        optimizer.zero_grad(set_to_none=True)

        # 计算梯度 (反向传播)
        loss.backward()

        # 4. 更新参数
        # 根据梯度更新模型参数
        optimizer.step()

    # ========================================================================
    # 训练完成
    # ========================================================================
    end_time = time.time()
    total_time = end_time - start_time

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"总耗时: {total_time:.2f}s ({total_time/60:.1f} 分钟)")
    print(f"平均速度: {cfg.max_iters / total_time:.1f} steps/s")

    # ========================================================================
    # 步骤 5: 保存模型
    # ========================================================================
    print("\n保存模型...")

    # 保存模型权重 (state_dict)
    # state_dict 包含所有可学习参数的值
    torch.save(model.state_dict(), 'minillm.pt')
    print("✓ 模型已保存至 minillm.pt")

    # 可选: 保存完整检查点 (包含优化器状态)
    # 这样可以从中断处恢复训练
    # checkpoint = {
    #     'model': model.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    #     'iter': cfg.max_iters,
    #     'config': model_cfg,
    # }
    # torch.save(checkpoint, 'checkpoint.pt')

    # ========================================================================
    # 训练总结
    # ========================================================================
    print("\n" + "=" * 60)
    print("训练总结")
    print("=" * 60)

    # 最终评估
    final_losses = estimate_loss(model, loader, cfg.eval_iters, device)
    print(f"最终训练损失:   {final_losses['train']:.4f}")
    print(f"最终验证损失:   {final_losses['val']:.4f}")
    print(f"过拟合程度:     {(final_losses['val'] - final_losses['train']):.4f}")

    if final_losses['val'] - final_losses['train'] > 0.5:
        print("\n⚠️  警告: 验证损失明显高于训练损失,可能存在过拟合")
        print("   建议: 增加正则化、减少模型大小或增加训练数据")
    elif final_losses['train'] > 3.0:
        print("\n⚠️  警告: 训练损失仍然较高,模型可能欠拟合")
        print("   建议: 增加模型大小、延长训练时间或调整学习率")
    else:
        print("\n✓ 训练看起来正常!")

    print("\n下一步:")
    print("  1. 运行 python generate.py 生成文本")
    print("  2. 调整超参数重新训练")
    print("  3. 尝试更大的模型或更多数据")


# ============================================================================
# 主入口
# ============================================================================

if __name__ == '__main__':
    """训练脚本入口

    运行方式:
    --------
    python train.py

    预期输出:
    --------
    1. 配置信息
    2. 数据加载进度
    3. 训练进度 (损失、时间)
    4. 模型保存信息
    5. 训练总结

    训练时长:
    --------
    - CPU: 约 10-30 分钟
    - GPU: 约 2-5 分钟
    - Apple Silicon (MPS): 约 5-10 分钟

    输出文件:
    --------
    - minillm.pt: 训练好的模型权重
    - tokenizer.model: 分词器模型 (如果之前不存在)
    - input_cn.txt: 训练数据 (如果之前不存在)

    故障排除:
    --------
    - 内存不足: 减小 batch_size 或 model.dim
    - 损失为 NaN: 降低学习率
    - 训练很慢: 确保使用 GPU (CUDA/MPS)
    """
    train()
