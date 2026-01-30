import torch
import torch.nn.functional as F
from config import ModelConfig
from model import MiniLLM
from tokenizer import Tokenizer

def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
    """
    给定上下文 `idx` 生成新 token。
    idx: (batch, seq_len) 索引张量
    """
    for _ in range(max_new_tokens):
        # 如果上下文太长，进行裁剪
        idx_cond = idx[:, -model.cfg.max_seq_len:]
        
        # 前向传播
        logits, _ = model(idx_cond)
        
        # 关注最后一个时间步
        logits = logits[:, -1, :] / temperature
        
        # 可选: Top-k 采样
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        # 可选: Top-p (Nucleus) 采样
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 移除累积概率超过阈值的 token
            sorted_indices_to_remove = cumulative_probs > top_p
            # 向右移动索引，以保留第一个超过阈值的 token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')

        # 应用 softmax 获取概率
        probs = F.softmax(logits, dim=-1)
        
        # 从分布中采样
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # 将采样到的索引拼接到序列中
        idx = torch.cat((idx, idx_next), dim=1)
        
    return idx

def main():
    # 1. 设置
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    cfg = ModelConfig()
    model = MiniLLM(cfg).to(device)
    
    # 2. 加载 Checkpoint (如果存在)
    try:
        model.load_state_dict(torch.load('minillm.pt', map_location=device))
        print("已加载 Checkpoint 'minillm.pt'")
    except FileNotFoundError:
        print("未找到 Checkpoint，使用随机初始化。")
        
    model.eval()
    
    # 3. 编码提示词 (Prompt)
    tokenizer = Tokenizer()
    prompt = "满纸荒唐言，" # 红楼梦开篇诗句
    start_ids = tokenizer.encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    
    # 4. 生成
    print(f"\n提示词: {prompt}")
    print("正在生成...")
    
    with torch.no_grad():
        y = generate(model, x, max_new_tokens=100, temperature=0.8, top_k=50)
        
    # 5. 解码
    generated_text = tokenizer.decode(y[0].tolist())
    print("\n--- 生成的文本 ---")
    print(generated_text)

if __name__ == '__main__':
    main()
