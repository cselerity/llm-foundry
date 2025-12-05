import torch
import time
from config import ModelConfig, TrainConfig
from model import MiniLLM
from data import DataLoader

def estimate_loss(model, loader, eval_iters, device):
    """估计训练集和验证集上的 loss。"""
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
    # 1. 设置
    cfg = TrainConfig()
    model_cfg = ModelConfig()
    
    # 自动检测设备
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 2. 数据与模型
    loader = DataLoader(batch_size=cfg.batch_size, block_size=model_cfg.max_seq_len, device=device)
    model = MiniLLM(model_cfg).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 3. 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    
    # 4. 训练循环
    print("开始训练...")
    start_time = time.time()
    
    for iter in range(cfg.max_iters):
        # 评估
        if iter % cfg.eval_interval == 0 or iter == cfg.max_iters - 1:
            losses = estimate_loss(model, loader, cfg.eval_iters, device)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
        # 采样 batch
        xb, yb = loader.get_batch('train')
        
        # 前向传播
        logits, loss = model(xb, yb)
        
        # 反向传播
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    end_time = time.time()
    print(f"训练完成，耗时 {end_time - start_time:.2f}s")
    
    # 5. 保存模型
    torch.save(model.state_dict(), 'minillm.pt')
    print("模型已保存至 minillm.pt")

if __name__ == '__main__':
    train()
