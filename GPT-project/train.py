import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import math
import os
import json
import time
from contextlib import nullcontext

# ────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────
@dataclass
class GPTConfig:
    block_size: int = 512
    batch_size: int = 12
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    vocab_size: int = 50257
    shared_expert_number: int = 4
    router_expert_number: int = 10
    top_k: int = 3

    @property
    def hidden_dim(self):
        return self.n_embd

    @property
    def head_size(self):
        return self.n_embd // self.n_head

    @property
    def d_ff(self):
        return self.hidden_dim * 2


# ────────────────────────────────────────────────────────────────────
# Model layers
# ────────────────────────────────────────────────────────────────────
class SingleHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key_layer = nn.Linear(config.hidden_dim, config.head_size)
        self.value_layer = nn.Linear(config.hidden_dim, config.head_size)
        self.query_layer = nn.Linear(config.hidden_dim, config.head_size)
        self.head_size = config.head_size
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "attention_mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
        )

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()
        k = self.key_layer(x)
        v = self.value_layer(x)
        q = self.query_layer(x)
        weight = q @ k.transpose(-2, -1)
        weight = weight.masked_fill(
            self.attention_mask[:seq_len, :seq_len] == 0,
            float("-inf")
        )
        weight = F.softmax(weight, dim=-1) / math.sqrt(self.head_size)
        weight = self.dropout(weight)
        return weight @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([
            SingleHeadAttention(config) for _ in range(config.n_head)
        ])
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        output = torch.cat([h(x) for h in self.heads], dim=-1)
        output = self.proj(output)
        return self.dropout(output)


class SwiGLUEExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_dim, config.d_ff)
        self.w2 = nn.Linear(config.d_ff, config.hidden_dim)
        self.w3 = nn.Linear(config.hidden_dim, config.d_ff)

    def forward(self, x):
        gate = F.silu(self.w1(x))
        activation = self.w3(x)
        return self.w2(gate * activation)


class MoERouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_dim, config.router_expert_number)
        self.top_k = config.top_k
        self.router_expert_number = config.router_expert_number

    def forward(self, x):
        router_logits = self.gate(x)
        router_prob = F.softmax(router_logits, dim=-1, dtype=torch.float)
        top_k_prob, select_expert_index = torch.topk(router_prob, k=self.top_k, dim=-1)
        top_k_prob = top_k_prob / top_k_prob.sum(dim=-1, keepdim=True)
        top_k_prob = top_k_prob.to(x.dtype)
        expert_mask = F.one_hot(
            select_expert_index,
            num_classes=self.router_expert_number,
        )
        expert_mask = expert_mask.permute(2, 1, 0)
        return router_logits, top_k_prob, select_expert_index, expert_mask


class SparseMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.router_expert_number = config.router_expert_number
        self.shared_expert_number = config.shared_expert_number
        self.top_k = config.top_k

        self.experts = nn.ModuleList([
            SwiGLUEExpert(config=config) for _ in range(self.router_expert_number)
        ])
        self.shared_experts = nn.ModuleList([
            SwiGLUEExpert(config=config) for _ in range(self.shared_expert_number)
        ])
        self.router = MoERouter(config=config)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()
        hidden_states = x.view(-1, hidden_dim)
        total_tokens = hidden_states.size(0)

        # 共享专家：所有 token 都通过
        shared_output = torch.zeros_like(hidden_states)
        for shared_expert in self.shared_experts:
            shared_output = shared_output + shared_expert(hidden_states)
        if self.shared_expert_number > 0:
            shared_output = shared_output / self.shared_expert_number

        # 路由专家
        router_logits, top_k_prob, select_expert_index, expert_mask = self.router(hidden_states)
        self._balance_loss_value = self.load_balance_loss(router_logits=router_logits, expert_mask=expert_mask)

        final_hidden_states = torch.zeros(
            (total_tokens, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        for expert_index in range(self.router_expert_number):
            expert_layer = self.experts[expert_index]
            current_expert_mask = expert_mask[expert_index]
            index, top_x = torch.where(current_expert_mask)
            if top_x.numel() == 0:
                continue

            current_state = hidden_states[top_x, :]
            current_state = expert_layer(current_state)
            current_token_router_weight = top_k_prob[top_x, index].unsqueeze(-1)
            current_hidden_states = current_state * current_token_router_weight
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states + shared_output
        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)
        return final_hidden_states, router_logits, self._balance_loss_value

    def load_balance_loss(self, router_logits, expert_mask):
        top1_mask = expert_mask[:, 0, :]
        fraction = top1_mask.float().mean(dim=-1)
        probs = F.softmax(router_logits, dim=-1)
        avg_prob = probs.mean(dim=0)
        return self.router_expert_number * torch.sum(fraction * avg_prob)


class Block_MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(config)
        self.ffn = SparseMoE(config)
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        final_hidden_states, _, _balance_loss_value = self.ffn(self.ln2(x))
        x = x + final_hidden_states
        self._balance_loss_value = _balance_loss_value
        return x


class GPT_MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.block_size = config.block_size
        self.block = nn.Sequential(*[Block_MoE(config) for _ in range(config.n_layer)])
        self.ln_final = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        batch, seq_len = idx.size()
        token_emb = self.tok_emb(idx)
        position_emb = self.pos_emb(torch.arange(seq_len, device=idx.device))
        x = token_emb + position_emb
        x = self.block(x)

        balance_loss = 0
        for block_item in self.block:
            balance_loss = balance_loss + block_item._balance_loss_value

        x = self.ln_final(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        else:
            batch, seq_len, vocab_size = logits.size()
            logits = logits.view(batch * seq_len, vocab_size)
            targets = targets.view(batch * seq_len)
            ce_loss = F.cross_entropy(logits, targets)
            return logits, ce_loss + balance_loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, path, block_size=512, max_lines=None):
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size
        self.eos_token = self.enc.encode(
            "<|endofttext|>", allowed_special={"<|endofttext|>"}
        )[0]
        self.encoded_data = []

        raw_data = []
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if max_lines is not None and i >= max_lines:
                    break
                try:
                    text = json.loads(line.strip())["text"]
                    raw_data.append(text)
                except Exception:
                    continue

        full_encoded = []
        for text in raw_data:
            encoded_text = self.enc.encode(text)
            full_encoded.extend(encoded_text + [self.eos_token])

        for i in range(0, len(full_encoded), self.block_size):
            chunk = full_encoded[i: i + self.block_size + 1]
            if len(chunk) < self.block_size + 1:
                chunk = chunk + [self.eos_token] * (self.block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def encode(self, text):
        return self.enc.encode(text)

    def decode(self, ids):
        return self.enc.decode(ids)


# ────────────────────────────────────────────────────────────────────
# Training utilities
# ────────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_grad_scaler(device, use_amp):
    if use_amp and device == "cuda":
        return torch.amp.GradScaler("cuda")
    return None


@torch.no_grad()
def estimate_loss(model, val_loader, device, ctx):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        with ctx:
            _, loss = model(x, targets=y)
        total_loss += loss.item() * x.size(0)
        total_tokens += x.size(0)
    model.train()
    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, val_loss, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        ckpt["scaler_state_dict"] = scaler.state_dict()
    torch.save(ckpt, os.path.join(save_dir, f"ckpt_epoch{epoch}_step{step}.pt"))
    print(f"[checkpoint] saved: epoch={epoch} step={step} val_loss={val_loss:.4f}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    start_step = ckpt["step"] + 1
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    print(f"[checkpoint] loaded from {path}, resuming at epoch={start_epoch} step={start_step}")
    return start_epoch, start_step


# ────────────────────────────────────────────────────────────────────
# Main training loop
# ────────────────────────────────────────────────────────────────────
def main():
    # ── hyperparameters ──
    data_path = "./mobvoi_seq_monkey_general_open_corpus.jsonl"
    max_data_lines = 1000
    batch_size = 4            # MoE 模型显存消耗大，酌情调小
    gradient_accumulation_steps = 4
    learning_rate = 3e-4
    weight_decay = 0.1
    betas = (0.9, 0.95)
    epochs = 10
    val_every_steps = 200
    save_every_epochs = 1
    generate_every_epochs = 1
    use_amp = True
    resume_from = None        # 设置 checkpoint 路径即可续训
    save_dir = "./checkpoints"

    # ── init ──
    device = get_device()
    config = GPTConfig()
    model = GPT_MoE(config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params / 1e6:.2f}M total, {trainable_params / 1e6:.2f}M trainable")
    print(f"Device: {device}")

    # ── data ──
    dataset = TextDataset(data_path, block_size=config.block_size, max_lines=max_data_lines)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=(device == "cuda"))
    print(f"Data: {len(train_dataset)} train samples, {len(val_dataset)} val samples, {len(train_loader)} train batches")

    # ── optimizer & scheduler ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas)
    total_steps = (len(train_loader) // gradient_accumulation_steps) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # ── amp ──
    scaler = get_grad_scaler(device, use_amp)
    amp_ctx = torch.amp.autocast("cuda") if (use_amp and device == "cuda") else nullcontext()

    start_epoch = 0
    global_step = 0

    if resume_from is not None:
        start_epoch, global_step = load_checkpoint(resume_from, model, optimizer, scheduler, scaler)

    # ── training ──
    model.train()
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        accumulated_loss = 0.0
        optimizer.zero_grad()
        t0 = time.time()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            with amp_ctx:
                _, loss = model(x, targets=y)
                loss = loss / gradient_accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_loss += loss.item() * gradient_accumulation_steps

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % val_every_steps == 0:
                    val_loss = estimate_loss(model, val_loader, device, amp_ctx)
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    tokens_per_sec = (batch_size * config.block_size * val_every_steps * gradient_accumulation_steps) / elapsed
                    print(
                        f"epoch {epoch:2d} | step {global_step:5d} | "
                        f"loss {accumulated_loss / val_every_steps:.4f} | "
                        f"val_loss {val_loss:.4f} | "
                        f"lr {lr:.2e} | "
                        f"tok/s {tokens_per_sec:.0f}"
                    )
                    accumulated_loss = 0.0
                    t0 = time.time()

            # 前几个 batch 打印 loss 方便观察
            if batch_idx < 3 or (batch_idx + 1) % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"  batch {batch_idx:4d} | loss {loss.item() * gradient_accumulation_steps:.4f} | lr {current_lr:.2e}")

        # ── end of epoch ──
        val_loss = estimate_loss(model, val_loader, device, amp_ctx)
        print(f"=== epoch {epoch} done | val_loss {val_loss:.4f} ===")

        if (epoch + 1) % save_every_epochs == 0:
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, val_loss, save_dir)

        # 生成几个 token 看效果
        if (epoch + 1) % generate_every_epochs == 0:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            prompt = "The capital of France is"
            prompt_ids = enc.encode(prompt)
            x_gen = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            out = model.generate(x_gen, max_new_tokens=30, temperature=0.8, top_k=50)
            generated = enc.decode(out[0].tolist())
            print(f"[generate] prompt: {prompt}")
            print(f"[generate] output: {generated}")
            print("───")

    print("Training complete.")


if __name__ == "__main__":
    main()
