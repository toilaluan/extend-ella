import torch
from torch import nn
from torch.optim import AdamW
from datasets import load_dataset
import random
from tqdm import tqdm
from transformers import GemmaTokenizerFast
from transformers.models.t5gemma.modeling_t5gemma import T5GemmaEncoder
from safetensors.torch import load_file, save_file
from ella.perceiver import ELLA, T5TextEmbedder
from ella.transformer import Block

class Config:
    dataset_size = 100000
    val_size = 1000
    batch_size = 1
    epochs = 10
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

class ExtendELLA(ELLA):
    pinned_timesteps: torch.Tensor
    def __init__(self, num_extend_blocks: int = 8, extend_mlp_ratio: int = 4, extend_num_attention_heads: int = 8, extend_head_dim: int = 64, pretrained_ella_path: str = None):
        super().__init__()
        if pretrained_ella_path is not None:
            print(f"Loading pretrained ella from: {pretrained_ella_path}")
            d = load_file(pretrained_ella_path)
            self.load_state_dict(d, strict=True)
            for p in self.parameters():
                p.requires_grad_(False)
            self.eval()
        self.e_hidden_size = extend_num_attention_heads * extend_head_dim
        self.pre_norm = nn.RMSNorm(self.width)
        self.e_proj_in = nn.Linear(self.width, self.e_hidden_size)
        self.e_blocks = nn.ModuleList(
            [
                Block(hidden_size=self.e_hidden_size, mlp_ratio=extend_mlp_ratio, num_attention_heads=extend_num_attention_heads, head_dim=extend_head_dim) for i in range(num_extend_blocks)
            ]
        )
        pinned_timesteps = torch.tensor([0.0])
        self.register_buffer("pinned_timesteps", pinned_timesteps)
       
    def forward(self, text_encode_features_list):
        all_hidden_states = []
        with torch.no_grad():
            for text_encode_features in text_encode_features_list:
                hidden_states = super().forward(text_encode_features, self.pinned_timesteps)
                all_hidden_states.append(hidden_states)
            hidden_states = torch.cat(all_hidden_states, dim=0)
        hidden_states = self.pre_norm(hidden_states)
        hidden_states = self.e_proj_in(hidden_states)
        for block in self.e_blocks:
            hidden_states = block(
                hidden_states
            )
        return hidden_states

class TeacherTextEmbedder:
    def __init__(self):
        self.tokenizer = GemmaTokenizerFast.from_pretrained("Photoroom/prx-1024-t2i-beta", subfolder="tokenizer")
        self.encoder = T5GemmaEncoder.from_pretrained("Photoroom/prx-1024-t2i-beta", subfolder="text_encoder", torch_dtype=torch.bfloat16).to(device)
        self.encoder.eval()

    def __call__(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        batch = self.tokenizer(prompts, max_length=Config.teacher_max_length, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            hidden_states = self.encoder(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
            embeds = hidden_states.float() * attn_mask.unsqueeze(-1)
            embeds = embeds.sum(dim=1)
            mask_sum = attn_mask.sum(dim=1, keepdim=True).clamp(min=1)
            targets = (embeds / mask_sum).to(torch.bfloat16)
            
        return targets


def drop_tokens(caption, prob):
    if prob == 0:
        return caption
    words = caption.split()
    kept = [w for w in words if random.random() >= prob]
    return " ".join(kept)

device = torch.device("cuda")

# Load dataset
ds = load_dataset("allenai/pixmo-cap", split="train")
ds = ds.shuffle(seed=Config.seed)
ds = ds.select(range(Config.dataset_size))
train_size = Config.dataset_size - Config.val_size
train_ds = ds.select(range(train_size)).with_format("torch")
val_ds = ds.select(range(train_size, Config.dataset_size)).with_format("torch")

teacher_embedder = TeacherTextEmbedder()
pre_ella_embedder = T5TextEmbedder(max_length=Config.max_length).to(device)

model = ExtendELLA(
    num_extend_blocks=Config.num_extend_blocks,
    extend_mlp_ratio=Config.extend_mlp_ratio,
    extend_num_attention_heads=Config.extend_num_attention_heads,
    extend_head_dim=Config.extend_head_dim,
    pretrained_ella_path=Config.pretrained_ella_path
).to(device)

# model = torch.compile(model)
# teacher_embedder = torch.compile(teacher_embedder)
# pre_ella_embedder = torch.compile(pre_ella_embedder)

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(trainable_params, lr=Config.lr)
mse = nn.MSELoss()

for epoch in range(Config.epochs):
    model.train()
    train_loss = 0.0
    train_ds = train_ds.shuffle()
    num_train_batches = (train_size + Config.batch_size - 1) // Config.batch_size
    for batch_idx in tqdm(range(num_train_batches), desc=f"Epoch {epoch+1}/{Config.epochs} - Train"):
        start = batch_idx * Config.batch_size
        end = min(start + Config.batch_size, train_size)
        captions = train_ds[start:end]["caption"]
        dropped = [drop_tokens(c, Config.token_dropout_prob) for c in captions]
        
        # Generate targets with Gemma
        targets = teacher_embedder(dropped)
        
        # Generate inputs with Flan
        list_hidden_states = pre_ella_embedder(dropped)
        
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(list_hidden_states)
            pred = out.mean(dim=1)
        loss = mse(pred, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Loss: {loss.item()}")
    
    avg_train_loss = train_loss / num_train_batches
    print(f"Average Train Loss: {avg_train_loss:.4f}")
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        num_val_batches = (Config.val_size + Config.batch_size - 1) // Config.batch_size
        for batch_idx in tqdm(range(num_val_batches), desc=f"Epoch {epoch+1}/{Config.epochs} - Val"):
            start = batch_idx * Config.batch_size
            end = min(start + Config.batch_size, Config.val_size)
            captions = val_ds[start:end]["caption"]
            
            targets = teacher_embedder(captions)
            
            # Generate inputs with Flan
            hidden_states, attn_mask = pre_ella_embedder(captions)
            text_encode_features_list = list(hidden_states.unbind(0))
            
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(text_encode_features_list)
                pred = out.mean(dim=1)
            loss = mse(pred, targets)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / num_val_batches
    print(f"Average Val Loss: {avg_val_loss:.4f}")

# Save the trained model
save_file(model.state_dict(), Config.save_path)