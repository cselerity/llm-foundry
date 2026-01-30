"""
Generate - 文本生成脚本
=====================

本模块实现了基于训练好的语言模型的文本生成功能,支持多种采样策略。

教学目标:
--------
- 理解自回归文本生成
- 掌握不同的采样策略 (贪心、温度、Top-k、Top-p)
- 学习如何控制生成质量和多样性
- 理解语言模型推理过程

核心概念:
--------
1. **自回归生成 (Autoregressive Generation)**:
   逐个生成 token,每个新 token 基于之前的所有 tokens

2. **采样策略 (Sampling Strategy)**:
   从概率分布中选择下一个 token 的方法
   - 贪心: 总是选择概率最高的
   - 随机采样: 按概率随机选择
   - Top-k: 只从概率最高的 k 个中选择
   - Top-p (Nucleus): 累积概率达到 p 的集合中选择

3. **温度 (Temperature)**:
   控制概率分布的"平滑度"
   - 低温 (0.5): 更确定,更保守
   - 高温 (1.5): 更随机,更有创意

生成流程:
--------
    Prompt (提示词)
        ↓ Tokenize
    初始 Token 序列
        ↓
    ┌─────────────────┐
    │  自回归循环     │  ← 重复 max_new_tokens 次
    ├─────────────────┤
    │ 1. 前向传播     │
    │ 2. 获取 logits  │
    │ 3. 应用温度     │
    │ 4. 采样策略     │
    │ 5. 选择下一token│
    │ 6. 拼接到序列   │
    └─────────────────┘
        ↓
    生成的 Token 序列
        ↓ Detokenize
    生成的文本

采样策略对比:
-----------
| 策略      | 优点             | 缺点             | 适用场景         |
|-----------|------------------|------------------|------------------|
| 贪心      | 确定性,流畅      | 重复,无创意      | 翻译、摘要       |
| 温度      | 可控随机性       | 可能不连贯       | 创意写作         |
| Top-k     | 平衡质量和多样性 | 固定 k 值不灵活  | 对话、故事生成   |
| Top-p     | 动态调整候选数   | 计算稍复杂       | 通用文本生成     |

与主工程的关系:
-------------
本文件是 src/llm_foundry/inference/ 的教学展示版本。
主工程支持批量生成、流式输出、KV 缓存等优化。
"""

import torch
import torch.nn.functional as F
from config import ModelConfig
from model import MiniLLM
from tokenizer import Tokenizer


# ============================================================================
# 文本生成函数
# ============================================================================

def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
    """自回归文本生成

    教学要点:
    -------
    - 逐个生成 token (自回归)
    - 每次只预测下一个 token
    - 支持多种采样策略
    - 自动处理上下文长度限制

    参数:
    ----
    model : MiniLLM
        训练好的语言模型
    idx : torch.Tensor
        输入上下文,形状 (batch_size, seq_len)
        通常 batch_size=1 (单个样本生成)
    max_new_tokens : int
        要生成的新 token 数量
    temperature : float
        采样温度,默认 1.0
        - < 1.0: 更确定,更保守 (如 0.5, 0.7)
        - = 1.0: 标准采样
        - > 1.0: 更随机,更有创意 (如 1.2, 1.5)
    top_k : int, optional
        Top-k 采样的 k 值
        只从概率最高的 k 个 token 中采样
        典型值: 40-50
    top_p : float, optional
        Top-p (Nucleus) 采样的 p 值
        只从累积概率达到 p 的 token 集合中采样
        典型值: 0.9-0.95

    返回:
    ----
    torch.Tensor
        生成的完整序列,形状 (batch_size, original_len + max_new_tokens)

    生成过程详解:
    -----------
    1. **上下文裁剪**: 如果序列超过 max_seq_len,只保留最后 max_seq_len 个 tokens
    2. **前向传播**: 获取下一个 token 的概率分布 (logits)
    3. **温度调整**: logits / temperature (温度越高,分布越平滑)
    4. **Top-k 过滤**: 保留概率最高的 k 个 tokens (可选)
    5. **Top-p 过滤**: 保留累积概率达到 p 的 tokens (可选)
    6. **Softmax**: 将 logits 转换为概率分布
    7. **采样**: 从概率分布中随机选择一个 token
    8. **拼接**: 将新 token 添加到序列末尾
    9. **重复**: 直到生成 max_new_tokens 个新 tokens

    温度的作用:
    ----------
    温度通过缩放 logits 来调整概率分布:

    - temperature = 1.0 (标准):
      P(token) = softmax(logits)

    - temperature < 1.0 (低温,更确定):
      高概率 token 的概率更高
      低概率 token 的概率更低
      生成更保守、更流畅,但可能重复

    - temperature > 1.0 (高温,更随机):
      概率分布更平滑
      低概率 token 也有更多机会被选中
      生成更有创意,但可能不连贯

    Top-k 采样:
    ----------
    只从概率最高的 k 个 token 中采样:
    1. 对 logits 排序,找到第 k 大的值
    2. 将小于第 k 大值的 logits 设为 -inf
    3. 这些 token 的概率会变为 0 (softmax 后)

    优点: 简单有效,避免选择极低概率的 token
    缺点: k 值固定,不适应不同的概率分布

    Top-p (Nucleus) 采样:
    -------------------
    只从累积概率达到 p 的最小 token 集合中采样:
    1. 按概率降序排列所有 tokens
    2. 计算累积概率
    3. 找到累积概率刚好超过 p 的位置
    4. 移除累积概率超过 p 的 tokens

    优点: 动态调整候选集大小,适应不同分布
    缺点: 计算稍复杂

    示例:
    ----
    >>> # 贪心生成 (temperature=0 相当于贪心)
    >>> y = generate(model, x, max_new_tokens=50, temperature=0.01)
    >>>
    >>> # 标准采样
    >>> y = generate(model, x, max_new_tokens=50, temperature=1.0)
    >>>
    >>> # Top-k 采样
    >>> y = generate(model, x, max_new_tokens=50, temperature=0.8, top_k=40)
    >>>
    >>> # Top-p 采样
    >>> y = generate(model, x, max_new_tokens=50, temperature=0.8, top_p=0.9)
    >>>
    >>> # 组合 Top-k 和 Top-p
    >>> y = generate(model, x, max_new_tokens=50, temperature=0.8, top_k=50, top_p=0.9)
    """
    for _ in range(max_new_tokens):
        # 步骤 1: 上下文裁剪
        # ----------------------------------------------------------------
        # 如果序列长度超过模型的最大长度,只保留最后 max_seq_len 个 tokens
        # 这是因为模型只能处理固定长度的序列 (由 RoPE 决定)
        idx_cond = idx[:, -model.cfg.max_seq_len:]

        # 步骤 2: 前向传播
        # ----------------------------------------------------------------
        # 获取下一个 token 的 logits (未归一化的概率)
        # logits 形状: (batch_size, seq_len, vocab_size)
        logits, _ = model(idx_cond)

        # 步骤 3: 提取最后一个时间步的 logits
        # ----------------------------------------------------------------
        # 我们只关心序列最后一个位置的预测
        # logits 形状变为: (batch_size, vocab_size)
        logits = logits[:, -1, :]

        # 应用温度
        # 温度越高,分布越平滑 (更随机)
        # 温度越低,分布越尖锐 (更确定)
        logits = logits / temperature

        # 步骤 4: Top-k 采样 (可选)
        # ----------------------------------------------------------------
        if top_k is not None:
            # 找到概率最高的 k 个 token
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))

            # 将概率低于第 k 个的 token 的 logits 设为 -inf
            # 这样它们的概率会变为 0 (softmax 后)
            logits[logits < v[:, [-1]]] = -float('Inf')

        # 步骤 5: Top-p (Nucleus) 采样 (可选)
        # ----------------------------------------------------------------
        if top_p is not None:
            # 按概率降序排列 logits
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)

            # 计算累积概率
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # 标记累积概率超过 p 的 tokens (要移除)
            sorted_indices_to_remove = cumulative_probs > top_p

            # 向右移动索引,保留第一个超过阈值的 token
            # 这确保至少有一个 token 被保留
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # 将排序后的标记映射回原始索引
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

            # 将要移除的 tokens 的 logits 设为 -inf
            logits[indices_to_remove] = -float('Inf')

        # 步骤 6: 转换为概率分布
        # ----------------------------------------------------------------
        # Softmax: 将 logits 转换为概率 (和为 1)
        probs = F.softmax(logits, dim=-1)

        # 步骤 7: 从分布中采样
        # ----------------------------------------------------------------
        # torch.multinomial: 多项式采样
        # 根据概率分布随机选择一个 token
        idx_next = torch.multinomial(probs, num_samples=1)

        # 步骤 8: 拼接新 token
        # ----------------------------------------------------------------
        # 将新生成的 token 添加到序列末尾
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


# ============================================================================
# 主函数
# ============================================================================

def main():
    """文本生成主流程

    教学要点:
    -------
    - 加载训练好的模型
    - 准备提示词 (Prompt)
    - 配置生成参数
    - 生成并展示文本

    步骤:
    ----
    1. 初始化模型和分词器
    2. 加载训练好的权重
    3. 编码提示词
    4. 生成新 tokens
    5. 解码为文本

    生成质量优化:
    -----------
    - 调整 temperature (0.7-1.0 通常效果好)
    - 使用 Top-k (40-50) 或 Top-p (0.9-0.95)
    - 提供更长的 Prompt (给模型更多上下文)
    - 训练更多轮次 (更好的模型 → 更好的生成)
    """

    # ========================================================================
    # 步骤 1: 初始化设备和模型
    # ========================================================================
    print("=" * 60)
    print("初始化文本生成")
    print("=" * 60)

    # 自动检测设备
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"使用设备: {device}\n")

    # 创建模型
    cfg = ModelConfig()
    model = MiniLLM(cfg).to(device)

    # ========================================================================
    # 步骤 2: 加载训练好的模型权重
    # ========================================================================
    print("加载模型权重...")
    try:
        model.load_state_dict(torch.load('minillm.pt', map_location=device))
        print("✓ 已加载 Checkpoint 'minillm.pt'\n")
    except FileNotFoundError:
        print("⚠️  未找到 Checkpoint 'minillm.pt'")
        print("   使用随机初始化的权重 (生成质量会很差)")
        print("   请先运行 'python train.py' 训练模型\n")

    # 切换到评估模式 (禁用 Dropout)
    model.eval()

    # ========================================================================
    # 步骤 3: 准备提示词 (Prompt)
    # ========================================================================
    print("准备提示词...")

    # 初始化分词器
    tokenizer = Tokenizer()

    # 提示词: 红楼梦开篇诗句的一部分
    # 模型会续写这段话
    prompt = "满纸荒唐言，"

    # 将提示词编码为 Token IDs
    start_ids = tokenizer.encode(prompt)

    # 转换为 PyTorch 张量并添加 batch 维度
    # 形状: (1, prompt_len)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    print(f"提示词: '{prompt}'")
    print(f"Token IDs: {start_ids}")
    print(f"Token 数量: {len(start_ids)}\n")

    # ========================================================================
    # 步骤 4: 配置生成参数
    # ========================================================================
    print("生成参数:")

    # 生成参数
    max_new_tokens = 100    # 生成 100 个新 tokens
    temperature = 0.8       # 温度 (0.8 = 略微保守)
    top_k = 50             # Top-k 采样 (k=50)
    top_p = None           # Top-p 采样 (不使用)

    print(f"  最大新 Tokens: {max_new_tokens}")
    print(f"  温度:          {temperature}")
    print(f"  Top-k:         {top_k if top_k else '不使用'}")
    print(f"  Top-p:         {top_p if top_p else '不使用'}\n")

    # ========================================================================
    # 步骤 5: 生成文本
    # ========================================================================
    print("─" * 60)
    print("正在生成...")
    print("─" * 60)

    # 禁用梯度计算 (推理时不需要梯度)
    with torch.no_grad():
        y = generate(
            model,
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

    print("✓ 生成完成!\n")

    # ========================================================================
    # 步骤 6: 解码并展示
    # ========================================================================
    print("=" * 60)
    print("生成的文本")
    print("=" * 60)

    # 解码: Token IDs → 文本
    generated_text = tokenizer.decode(y[0].tolist())

    print(generated_text)
    print("\n" + "=" * 60)

    # ========================================================================
    # 统计信息
    # ========================================================================
    print("\n生成统计:")
    print(f"  原始提示长度:   {len(start_ids)} tokens")
    print(f"  生成的 tokens:   {max_new_tokens} tokens")
    print(f"  总长度:          {len(y[0])} tokens")
    print(f"  文本长度:        {len(generated_text)} 字符")

    # ========================================================================
    # 提示: 尝试不同的参数
    # ========================================================================
    print("\n" + "=" * 60)
    print("尝试不同的生成参数")
    print("=" * 60)
    print("在代码中修改以下参数,观察生成效果:")
    print("")
    print("1. temperature (温度):")
    print("   - 0.5: 更保守,更流畅 (适合正式文本)")
    print("   - 1.0: 标准采样 (平衡)")
    print("   - 1.5: 更有创意 (适合创意写作)")
    print("")
    print("2. top_k (Top-k 采样):")
    print("   - 20: 非常保守")
    print("   - 50: 平衡 (推荐)")
    print("   - 100: 更多样化")
    print("")
    print("3. top_p (Nucleus 采样):")
    print("   - 0.9: 保守")
    print("   - 0.95: 平衡 (推荐)")
    print("   - 0.99: 更多样化")
    print("")
    print("4. max_new_tokens (生成长度):")
    print("   - 50: 短文本")
    print("   - 100: 中等长度 (当前)")
    print("   - 200+: 长文本")


# ============================================================================
# 主入口
# ============================================================================

if __name__ == '__main__':
    """文本生成脚本入口

    运行方式:
    --------
    python generate.py

    前提条件:
    --------
    1. 已训练模型: minillm.pt 存在
    2. 已训练分词器: tokenizer.model 存在

    预期输出:
    --------
    1. 初始化信息
    2. 提示词内容
    3. 生成的完整文本
    4. 统计信息
    5. 参数调整建议

    自定义生成:
    ----------
    修改 main() 函数中的以下变量:
    - prompt: 提示词内容
    - max_new_tokens: 生成长度
    - temperature: 采样温度
    - top_k: Top-k 参数
    - top_p: Top-p 参数

    示例提示词:
    ----------
    - "红楼梦是"
    - "今天天气"
    - "人工智能"
    - "从前有座山，"

    生成质量:
    --------
    生成质量取决于:
    1. 模型训练程度 (训练轮次、损失值)
    2. 提示词质量 (相关性、长度)
    3. 采样参数 (temperature, top_k, top_p)
    4. 训练数据质量
    """
    main()
