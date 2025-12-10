from transformers import GemmaTokenizerFast
from transformers.models.t5gemma.modeling_t5gemma import T5GemmaEncoder
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

device = "cuda"
text_encoder = T5GemmaEncoder.from_pretrained("Photoroom/prx-1024-t2i-beta", subfolder="text_encoder", dtype=torch.bfloat16).to(device)
tokenizer = GemmaTokenizerFast.from_pretrained("Photoroom/prx-1024-t2i-beta", subfolder="tokenizer")

ds = load_dataset("allenai/pixmo-cap", split="train").select_columns(["caption"])
ds = ds.shuffle(seed=42)
ds = ds.select(range(100000))

ds = ds.map(lambda x: tokenizer(x["caption"], max_length=256, padding="max_length", truncation=True), num_proc=8)
ds = ds.with_format("torch")
dl = DataLoader(ds, batch_size=2)

os.makedirs("embeddings", exist_ok=True)

id_counter = 0
for batch in tqdm(dl):
    with torch.no_grad():
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        embeddings = text_encoder(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
        )["last_hidden_state"]
        embeddings = embeddings * attn_mask[:, :, None]
        embeddings = embeddings.float().sum(dim=1)
        mask_sum = attn_mask.sum(dim=1, keepdim=True).clamp(min=1)
        embeddings = (embeddings / mask_sum).to(torch.bfloat16)
    
    captions = batch["caption"]
    for emb, cap in zip(embeddings, captions):
        torch.save(emb, f"embeddings/{id_counter}.pt")
        with open(f"embeddings/{id_counter}.txt", "w") as f:
            f.write(cap)
        id_counter += 1