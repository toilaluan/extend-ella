import torch
from torch import nn
from torch.optim import AdamW
import random
from tqdm import tqdm
from transformers import GemmaTokenizerFast
from transformers.models.t5gemma.modeling_t5gemma import T5GemmaEncoder
from safetensors.torch import load_file, save_file
from ella.perceiver import ELLA, T5TextEmbedder
from ella.transformer import Block

from prettytable import PrettyTable
import os
import wandb

wandb.init("toilaluan", project="extend-ella")


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


class Config:
    dataset_size = 50000
    val_size = 1000
    batch_size = 2
    epochs = 1
    lr = 1e-4
    token_dropout_prob = 0.5
    teacher_max_length = 256
    max_length = 256
    pretrained_ella_path = "ella-sd1.5-tsc-t5xl.safetensors"  # Set to the path of pretrained ELLA safetensors file
    num_extend_blocks = 8
    extend_mlp_ratio = 4
    extend_num_attention_heads = 36
    extend_head_dim = 64
    seed = 42
    save_path = "trained_extended_ella.safetensors"
    device = "cuda"


class ExtendELLA(ELLA):
    pinned_timesteps: torch.Tensor

    def __init__(
        self,
        num_extend_blocks: int = 8,
        extend_mlp_ratio: int = 4,
        extend_num_attention_heads: int = 8,
        extend_head_dim: int = 64,
        pretrained_ella_path: str = None,
        num_extend_visions: int = 256,
        vision_hidden_size: int = 1024,
    ):
        super().__init__()
        if pretrained_ella_path is not None:
            print(f"Loading pretrained ella from: {pretrained_ella_path}")
            d = load_file(pretrained_ella_path)
            self.load_state_dict(d, strict=True)
            for p in self.parameters():
                p.requires_grad_(False)
        self.e_hidden_size = extend_num_attention_heads * extend_head_dim
        self.pre_norm = nn.RMSNorm(self.width)
        self.e_proj_in = nn.Linear(self.width, self.e_hidden_size)
        self.e_blocks = nn.ModuleList(
            [
                Block(
                    hidden_size=self.e_hidden_size,
                    mlp_ratio=extend_mlp_ratio,
                    num_attention_heads=extend_num_attention_heads,
                    head_dim=extend_head_dim,
                )
                for i in range(num_extend_blocks)
            ]
        )
        self.vision_tokens = nn.Parameter(
            self.e_hidden_size**-0.5
            * torch.randn(num_extend_visions, self.e_hidden_size)
        )
        self.vision_norm = nn.RMSNorm(self.e_hidden_size)
        self.vision_proj = nn.Linear(self.e_hidden_size, vision_hidden_size)
        pinned_timesteps = torch.tensor([0.0])
        self.register_buffer("pinned_timesteps", pinned_timesteps)

    @torch.compiler.disable
    @torch.no_grad
    def get_ella_hidden_states(self, text_encode_features_list):
        all_hidden_states = []
        for text_encode_features in text_encode_features_list:
            hidden_states = super().forward(
                text_encode_features, self.pinned_timesteps
            )
            all_hidden_states.append(hidden_states)
        hidden_states = torch.cat(all_hidden_states, dim=0)
        return hidden_states

    def forward(self, text_encode_features_list):
        hidden_states = self.get_ella_hidden_states(text_encode_features_list)
        hidden_states = self.pre_norm(hidden_states)
        hidden_states = self.e_proj_in(hidden_states)
        B = hidden_states.shape[0]
        hidden_states = torch.cat(
            [hidden_states, self.visual_tokens.expand(B, 1, 1)], dim=1
        )
        for block in self.e_blocks:
            hidden_states = block(hidden_states)
        text_states = hidden_states[:-self.num_extend_visions]
        vision_states = hidden_states[-self.num_extend_visions:]
        vision_states = self.vision_norm(vision_states)
        vision_states = self.vision_proj(vision_states)
        return text_states, vision_states


def drop_tokens(caption, prob):
    if prob == 0:
        return caption
    words = caption.split()
    kept = [w for w in words if random.random() >= prob]
    return " ".join(kept)


device = torch.device(Config.device)

# Load from cached folder
cached_dir = "embeddings"
captions = []
targets = []
for i in tqdm(range(Config.dataset_size), desc="Loading cached data"):
    caption_path = os.path.join(cached_dir, f"{i}.txt")
    target_path = os.path.join(cached_dir, f"{i}.pt")
    if not os.path.exists(caption_path):
        continue
    with open(caption_path, "r") as f:
        captions.append(f.read().strip())
    targets.append(torch.load(target_path, map_location=device))

train_size = Config.dataset_size - Config.val_size
train_captions = captions[:train_size]
train_targets = targets[:train_size]
val_captions = captions[train_size:]
val_targets = targets[train_size:]

pre_ella_embedder = T5TextEmbedder(max_length=Config.max_length).to(device)

model = ExtendELLA(
    num_extend_blocks=Config.num_extend_blocks,
    extend_mlp_ratio=Config.extend_mlp_ratio,
    extend_num_attention_heads=Config.extend_num_attention_heads,
    extend_head_dim=Config.extend_head_dim,
    pretrained_ella_path=Config.pretrained_ella_path,
).to(device)

count_parameters(model)

model = torch.compile(model)
pre_ella_embedder = torch.compile(pre_ella_embedder)

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(trainable_params, lr=Config.lr)
mse = nn.MSELoss()

scaler = torch.amp.grad_scaler.GradScaler(device=Config.device)

random.seed(Config.seed)

for epoch in range(Config.epochs):
    model.train()
    train_loss = 0.0
    train_indices = list(range(train_size))
    random.shuffle(train_indices)
    num_train_batches = (train_size + Config.batch_size - 1) // Config.batch_size
    for batch_idx in tqdm(
        range(num_train_batches), desc=f"Epoch {epoch + 1}/{Config.epochs} - Train"
    ):
        start = batch_idx * Config.batch_size
        end = min(start + Config.batch_size, train_size)
        batch_indices = train_indices[start:end]
        batch_captions = [train_captions[j] for j in batch_indices]
        batch_targets = torch.stack([train_targets[j] for j in batch_indices])

        dropped = [drop_tokens(c, Config.token_dropout_prob) for c in batch_captions]
        input_ids, attention_mask = pre_ella_embedder.tokenize(dropped)
        # Generate inputs with Flan
        embeddings, attn_mask = pre_ella_embedder(input_ids, attention_mask)

        list_hidden_states = []
        for emb, mask in zip(embeddings.unbind(0), attn_mask.unbind(0)):
            list_hidden_states.append(emb[: mask.sum()].unsqueeze(0))

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(list_hidden_states)
            pred = out.mean(dim=1)
            loss = mse(pred, batch_targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Loss: {loss.item()}")
            wandb.log({"train_loss": loss.item()})

        if batch_idx % 5000 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                num_val_batches = (
                    Config.val_size + Config.batch_size - 1
                ) // Config.batch_size
                for batch_idx in tqdm(
                    range(num_val_batches),
                    desc=f"Epoch {epoch + 1}/{Config.epochs} - Val",
                ):
                    start = batch_idx * Config.batch_size
                    end = min(start + Config.batch_size, Config.val_size)
                    batch_captions = val_captions[start:end]
                    batch_targets = torch.stack(val_targets[start:end])
                    input_ids, attention_mask = pre_ella_embedder.tokenize(dropped)
                    # Generate inputs with Flan
                    embeddings, attn_mask = pre_ella_embedder(input_ids, attention_mask)

                    list_hidden_states = []
                    for emb, mask in zip(embeddings.unbind(0), attn_mask.unbind(0)):
                        list_hidden_states.append(emb[: mask.sum()].unsqueeze(0))

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = model(list_hidden_states)
                        pred = out.mean(dim=1)
                        loss = mse(pred, batch_targets)
                        val_loss += loss.item()
            avg_val_loss = val_loss / num_val_batches
            wandb.log({"val_loss": avg_val_loss})
            print(f"Average Val Loss: {avg_val_loss:.4f}")
            save_file(model.state_dict(), Config.save_path)

# Save the trained model
