import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (均方根层归一化)。
    通过使用均方根对输入进行归一化来稳定训练。
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (batch, seq_len, dim)
        # 计算 RMS: sqrt(mean(x^2))
        # 我们在 sqrt 内部加上 eps 以保证数值稳定性
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        return self.weight * x_norm

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    预计算 RoPE 中使用的复数指数 (cis) 频率张量。
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # (end, dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """
    对 query 和 key 应用旋转位置编码 (RoPE)。
    xq, xk: (batch, seq_len, n_heads, head_dim)
    freqs_cis: (seq_len, head_dim // 2)
    """
    # 重塑为复数: (batch, seq_len, n_heads, head_dim // 2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 广播 freqs_cis 以匹配 batch 和 heads
    # freqs_cis: (seq_len, head_dim // 2) -> (1, seq_len, 1, head_dim // 2)
    freqs_cis = freqs_cis.view(1, xq_.size(1), 1, xq_.size(-1))
    
    # 旋转
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.dim = cfg.dim
        self.head_dim = cfg.dim // cfg.n_heads
        
        # 投影层
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        
        self.dropout = nn.Dropout(cfg.dropout)
        
        # KV Cache 支持 (暂留占位符)
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, freqs_cis):
        batch_size, seq_len, _ = x.shape
        
        # (B, S, D) -> (B, S, H, D_h)
        xq = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # 应用 RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis[:seq_len])
        
        # 如果 n_kv_heads < n_heads (GQA)，则重复 KV heads
        # 我们需要重复 keys 和 values 以匹配 query heads 的数量
        if self.n_kv_heads != self.n_heads:
            xk = torch.repeat_interleave(xk, self.n_heads // self.n_kv_heads, dim=2)
            xv = torch.repeat_interleave(xv, self.n_heads // self.n_kv_heads, dim=2)
            
        # 转置以进行注意力计算: (B, H, S, D_h)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # 缩放点积注意力 (Scaled Dot-Product Attention)
        # 为了清晰起见，这里展示手动实现，但 F.scaled_dot_product_attention 更快
        # scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        # scores = scores + mask
        # probs = F.softmax(scores, dim=-1)
        # probs = self.dropout(probs)
        # output = torch.matmul(probs, xv)
        
        # 使用 PyTorch 2.0+ 的高效实现
        output = F.scaled_dot_product_attention(
            xq, xk, xv, 
            attn_mask=None, 
            dropout_p=self.dropout.p if self.training else 0.0, 
            is_causal=True
        )
        
        # (B, H, S, D_h) -> (B, S, H, D_h) -> (B, S, D)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)

class MLP(nn.Module):
    """
    使用 SwiGLU 激活函数的前馈网络 (FeedForward Network)。
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden_dim = 4 * cfg.dim
        hidden_dim = int(2 * hidden_dim / 3) # SwiGLU 惯例
        # 确保 hidden_dim 是 256 的倍数 (可选优化)
        
        self.w1 = nn.Linear(cfg.dim, hidden_dim, bias=False) # Gate (门控)
        self.w2 = nn.Linear(hidden_dim, cfg.dim, bias=False) # Down (降维)
        self.w3 = nn.Linear(cfg.dim, hidden_dim, bias=False) # Up (升维)
        
    def forward(self, x):
        # SwiGLU: (Swish(xW1) * xW3) W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attention = CausalSelfAttention(cfg)
        self.feed_forward = MLP(cfg)
        self.attention_norm = RMSNorm(cfg.dim)
        self.ffn_norm = RMSNorm(cfg.dim)
        
    def forward(self, x, freqs_cis):
        # Pre-normalization (前置归一化)
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class MiniLLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        
        # 权重绑定 (可选，但在 GPT 中很常见)
        # self.token_embedding.weight = self.output.weight 
        
        # 预计算 RoPE 频率
        self.freqs_cis = precompute_freqs_cis(
            self.cfg.dim // self.cfg.n_heads, 
            self.cfg.max_seq_len * 2 # *2 缓冲区
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, targets=None):
        batch_size, seq_len = tokens.shape
        h = self.token_embedding(tokens)
        
        # 获取当前序列长度的 RoPE 频率
        freqs_cis = self.freqs_cis[:seq_len].to(h.device)
        
        for layer in self.layers:
            h = layer(h, freqs_cis)
            
        h = self.norm(h)
        logits = self.output(h)
        
        loss = None
        if targets is not None:
            # 展平以进行 CrossEntropyLoss 计算
            # logits: (B, S, V) -> (B*S, V)
            # targets: (B, S) -> (B*S)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

if __name__ == '__main__':
    # 测试模型
    cfg = ModelConfig()
    model = MiniLLM(cfg)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    x = torch.randint(0, cfg.vocab_size, (2, 16)) # Batch=2, Seq=16
    logits, loss = model(x, x) # 虚拟目标
    print(f"Logits 形状: {logits.shape}")
    print(f"Loss: {loss.item()}")
