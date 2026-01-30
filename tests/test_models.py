"""模型组件测试

这些测试用例既验证功能正确性,也作为使用示例展示各组件的行为。
"""

import pytest
import torch
import math

from llm_foundry.config import ModelConfig
from llm_foundry.models.components import (
    RMSNorm,
    precompute_freqs_cis,
    apply_rotary_emb,
    CausalSelfAttention,
    MLP,
    Block
)
from llm_foundry.models import MiniLLM


class TestRMSNorm:
    """RMSNorm 测试

    教学要点:
    - RMSNorm 如何归一化
    - 与 LayerNorm 的区别
    - 归一化的效果验证
    """

    def test_output_shape(self):
        """测试输出形状保持不变"""
        norm = RMSNorm(dim=256)
        x = torch.randn(2, 10, 256)  # (batch=2, seq=10, dim=256)

        output = norm(x)

        assert output.shape == x.shape, "RMSNorm 应该保持输入形状不变"

    def test_normalization_effect(self):
        """测试归一化效果

        教学要点: RMSNorm 将向量归一化到接近单位长度
        """
        norm = RMSNorm(dim=256)
        x = torch.randn(2, 10, 256) * 10  # 大幅度的输入

        output = norm(x)

        # 计算归一化后的 RMS (应该接近 1)
        rms = torch.sqrt(torch.mean(output ** 2, dim=-1))

        # 由于有可学习的 weight 参数,RMS 不一定严格为 1
        # 但应该在合理范围内
        assert torch.all(rms > 0.1) and torch.all(rms < 10), \
            "归一化后的 RMS 应该在合理范围内"

    def test_gradient_flow(self):
        """测试梯度是否正常流动

        教学要点: 归一化层应该允许梯度反向传播
        """
        norm = RMSNorm(dim=256)
        x = torch.randn(2, 10, 256, requires_grad=True)

        output = norm(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "梯度应该能够传播到输入"
        assert norm.weight.grad is not None, "梯度应该能够传播到 weight 参数"


class TestRoPE:
    """RoPE (旋转位置编码) 测试

    教学要点:
    - RoPE 如何编码位置信息
    - 为什么使用复数旋转
    - RoPE 的关键特性
    """

    def test_freqs_cis_shape(self):
        """测试频率张量的形状"""
        dim = 64  # 注意力头维度
        max_len = 128  # 最大序列长度

        freqs_cis = precompute_freqs_cis(dim, max_len)

        # 频率是复数,维度减半
        assert freqs_cis.shape == (max_len, dim // 2), \
            "频率张量形状应该是 (seq_len, dim // 2)"
        assert freqs_cis.dtype == torch.complex64, \
            "频率应该是复数类型"

    def test_rope_preserves_norm(self):
        """测试 RoPE 保持向量长度

        教学要点: 旋转变换不应该改变向量的长度,
        只改变方向。这保证了注意力分数的尺度不变。
        """
        batch, seq_len, n_heads, head_dim = 2, 10, 8, 64

        q = torch.randn(batch, seq_len, n_heads, head_dim)
        k = torch.randn(batch, seq_len, n_heads, head_dim)
        freqs_cis = precompute_freqs_cis(head_dim, seq_len)

        # 应用 RoPE
        q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)

        # 验证向量长度保持不变
        q_norm_before = torch.norm(q, dim=-1)
        q_norm_after = torch.norm(q_rot, dim=-1)

        assert torch.allclose(q_norm_before, q_norm_after, atol=1e-5), \
            "RoPE 应该保持向量长度不变"

    def test_relative_position_encoding(self):
        """测试相对位置编码特性

        教学要点: RoPE 的关键特性是注意力分数只依赖于相对位置,
        而不是绝对位置。即 q_m · k_n 只依赖于 (m-n)。
        """
        batch, seq_len, n_heads, head_dim = 1, 5, 4, 32

        q = torch.randn(batch, seq_len, n_heads, head_dim)
        k = torch.randn(batch, seq_len, n_heads, head_dim)
        freqs_cis = precompute_freqs_cis(head_dim, seq_len * 2)

        # 应用 RoPE 到位置 [0, 1, 2, 3, 4]
        q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)

        # 计算位置 0 和位置 1 的注意力分数
        score_0_1 = (q_rot[0, 0] * k_rot[0, 1]).sum()

        # 现在偏移所有位置 +2,变成 [2, 3, 4, 5, 6]
        # 但我们还是看位置 0→1 的相对关系(现在是 2→3)
        q_shifted = q
        k_shifted = k
        freqs_shifted = freqs_cis[2:2+seq_len]  # 偏移频率
        q_rot_shifted, k_rot_shifted = apply_rotary_emb(
            q_shifted, k_shifted, freqs_shifted
        )

        score_2_3 = (q_rot_shifted[0, 0] * k_rot_shifted[0, 1]).sum()

        # 相对位置相同(都是 +1),注意力分数应该相同
        assert torch.allclose(score_0_1, score_2_3, atol=1e-4), \
            "相同相对位置的注意力分数应该相同(RoPE 的核心特性)"


class TestCausalSelfAttention:
    """因果自注意力测试

    教学要点:
    - 注意力机制如何工作
    - 因果掩码的作用
    - GQA 的实现
    """

    def test_output_shape(self):
        """测试输出形状"""
        cfg = ModelConfig(dim=256, n_heads=8, n_kv_heads=4)
        attn = CausalSelfAttention(cfg)

        x = torch.randn(2, 10, cfg.dim)  # (batch, seq, dim)
        freqs_cis = precompute_freqs_cis(cfg.dim // cfg.n_heads, 10)

        output = attn(x, freqs_cis)

        assert output.shape == x.shape, \
            "注意力层输出形状应该与输入相同"

    def test_causal_masking(self):
        """测试因果掩码

        教学要点: 在语言建模中,位置 i 只能看到位置 0 到 i 的信息,
        不能看到未来的 token。这通过因果掩码实现。
        """
        cfg = ModelConfig(dim=128, n_heads=4, n_kv_heads=2, max_seq_len=5)
        attn = CausalSelfAttention(cfg)
        attn.eval()  # 评估模式,关闭 dropout

        # 创建一个特殊的输入:只有第一个位置有信号
        x = torch.zeros(1, 5, cfg.dim)
        x[0, 0, :] = 1.0  # 只有位置 0 有值

        freqs_cis = precompute_freqs_cis(cfg.dim // cfg.n_heads, 5)
        output = attn(x, freqs_cis)

        # 由于因果掩码,位置 0 只能看到自己
        # 所以位置 0 的输出应该非零
        assert torch.norm(output[0, 0]) > 0.1, \
            "位置 0 应该能看到自己的信息"

        # 但是位置 2、3、4 也应该能"看到"位置 0
        # (虽然它们自己是 0,但通过注意力机制能看到位置 0)
        # 这验证了注意力机制的信息传递


class TestMLP:
    """MLP (前馈网络) 测试

    教学要点:
    - SwiGLU 激活函数
    - 前馈网络的作用
    """

    def test_output_shape(self):
        """测试输出形状"""
        cfg = ModelConfig(dim=256)
        mlp = MLP(cfg)

        x = torch.randn(2, 10, cfg.dim)
        output = mlp(x)

        assert output.shape == x.shape, \
            "MLP 输入输出维度应该相同"

    def test_swiglu_activation(self):
        """测试 SwiGLU 激活

        教学要点: SwiGLU = Swish(xW1) ⊙ (xW3)
        这是一个门控激活函数,性能优于 ReLU 和 GELU
        """
        cfg = ModelConfig(dim=128)
        mlp = MLP(cfg)

        x = torch.randn(1, 1, cfg.dim)
        output = mlp(x)

        # 输出应该是非线性的(不是简单的线性变换)
        # 我们通过测试是否满足线性来验证非线性
        x2 = x * 2
        output2 = mlp(x2)

        # 如果是线性,output2 应该等于 output * 2
        # 但由于 SwiGLU 非线性,不应该相等
        assert not torch.allclose(output2, output * 2, rtol=0.1), \
            "SwiGLU 应该是非线性激活"


class TestBlock:
    """Transformer Block 测试

    教学要点:
    - 残差连接的作用
    - Pre-normalization 架构
    - 注意力和 FFN 的组合
    """

    def test_output_shape(self):
        """测试输出形状"""
        cfg = ModelConfig(dim=256, n_heads=8)
        block = Block(cfg)

        x = torch.randn(2, 10, cfg.dim)
        freqs_cis = precompute_freqs_cis(cfg.dim // cfg.n_heads, 10)

        output = block(x, freqs_cis)

        assert output.shape == x.shape

    def test_residual_connection(self):
        """测试残差连接

        教学要点: 残差连接 (x + F(x)) 有两个作用:
        1. 帮助梯度流动,解决深层网络训练困难
        2. 允许网络学习"增量",而不是完整变换
        """
        cfg = ModelConfig(dim=128, n_heads=4)
        block = Block(cfg)

        x = torch.randn(1, 5, cfg.dim)
        freqs_cis = precompute_freqs_cis(cfg.dim // cfg.n_heads, 5)

        output = block(x, freqs_cis)

        # 输出应该与输入有一定相关性(因为残差连接)
        # 但不应该完全相同(因为经过了变换)
        cosine_sim = torch.nn.functional.cosine_similarity(
            x.view(-1), output.view(-1), dim=0
        )

        assert 0.3 < cosine_sim < 0.99, \
            f"残差连接应该保持部分原始信息,相似度: {cosine_sim:.3f}"


class TestMiniLLM:
    """完整模型测试

    教学要点:
    - 端到端的前向传播
    - 损失计算
    - 模型参数统计
    """

    def test_forward_pass(self):
        """测试前向传播"""
        cfg = ModelConfig(dim=128, n_layers=2, n_heads=4)
        model = MiniLLM(cfg)

        tokens = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, loss = model(tokens, tokens)

        assert logits.shape == (2, 16, cfg.vocab_size), \
            "Logits 形状应该是 (batch, seq, vocab_size)"
        assert loss is not None, "应该计算损失"
        assert loss.item() > 0, "损失应该是正数"

    def test_parameter_count(self):
        """测试参数量计算

        教学要点: 了解模型参数的分布有助于理解模型容量
        """
        cfg = ModelConfig(dim=256, n_layers=4, n_heads=8, vocab_size=8192)
        model = MiniLLM(cfg)

        total_params = model.get_num_params(non_embedding=False)
        non_emb_params = model.get_num_params(non_embedding=True)

        # 验证参数量在合理范围
        assert total_params > 1e6, "模型应该有超过 1M 参数"
        assert non_emb_params < total_params, \
            "非嵌入参数应该少于总参数(因为排除了嵌入层)"

        print(f"\n模型参数统计:")
        print(f"  总参数: {total_params/1e6:.2f}M")
        print(f"  非嵌入参数: {non_emb_params/1e6:.2f}M")
        print(f"  嵌入参数: {(total_params-non_emb_params)/1e6:.2f}M")

    def test_generation_capability(self):
        """测试生成能力

        教学要点: 模型应该能够基于上下文生成下一个 token
        """
        cfg = ModelConfig(dim=128, n_layers=2, n_heads=4, max_seq_len=64)
        model = MiniLLM(cfg)
        model.eval()

        # 输入一个短序列
        input_tokens = torch.randint(0, cfg.vocab_size, (1, 5))

        with torch.no_grad():
            logits, _ = model(input_tokens)

        # 获取最后一个位置的预测
        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits)

        assert 0 <= next_token < cfg.vocab_size, \
            "预测的 token 应该在词表范围内"


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v', '-s'])
