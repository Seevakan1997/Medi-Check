"""
"""

import os, torch, pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

CSV_PATH   = "data/health_fact.csv"

MODEL_NAME = "distilbert-base-uncased"

SAVE_PATH  = "model.pt"
EPOCHS     = 4        
BATCH      = 32       
LR         = 2e-5
MAX_LEN    = 128      
SAMPLE     = None     
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


class ClaimDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding=True,
            max_length=MAX_LEN, return_tensors="pt"
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.encodings.items()}, self.labels[i]

class MisinfoClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(MODEL_NAME)
        hidden          = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)


def train():
    print(f"Device : {DEVICE}")
    print(f"Model  : {MODEL_NAME}")
    print(f"Sample : {SAMPLE if SAMPLE else 'full dataset'}")
    print(f"MaxLen : {MAX_LEN}  |  Batch: {BATCH}  |  Epochs: {EPOCHS}\n")

    df = pd.read_csv(CSV_PATH).dropna(subset=["text", "label"])

    if SAMPLE:
        df = df.sample(n=min(SAMPLE, len(df)), random_state=42).reset_index(drop=True)

    train_df, test_df = train_test_split(df, test_size=0.15,
                                          stratify=df["label"], random_state=42)
    train_df, val_df  = train_test_split(train_df, test_size=0.15,
                                          stratify=train_df["label"], random_state=42)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    tokenizer    = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader = DataLoader(
        ClaimDataset(train_df["text"], train_df["label"], tokenizer),
        batch_size=BATCH, shuffle=True
    )
    val_loader   = DataLoader(
        ClaimDataset(val_df["text"], val_df["label"], tokenizer),
        batch_size=BATCH
    )
    test_loader  = DataLoader(
        ClaimDataset(test_df["text"], test_df["label"], tokenizer),
        batch_size=BATCH
    )

    model     = MisinfoClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader) // 10,
        num_training_steps=len(train_loader) * EPOCHS
    )

    best_val_loss = float("inf")
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        total_loss = 0
        for batch, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
            batch  = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = labels.to(DEVICE)
            logits = model(**batch)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch, labels in val_loader:
                batch  = {k: v.to(DEVICE) for k, v in batch.items()}
                labels = labels.to(DEVICE)
                val_loss += criterion(model(**batch), labels).item()

        avg_val = val_loss / len(val_loader)
        print(f"  train loss: {total_loss/len(train_loader):.4f} | val loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✓ Best model saved → {SAVE_PATH}")

    print("\n── Test Results ──────────────────────────────")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for batch, labels in test_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            preds.extend(model(**batch).argmax(-1).cpu().tolist())
            actuals.extend(labels.tolist())

    print(classification_report(actuals, preds,
                                  target_names=["credible", "misinformation"]))
    print(f"Done. Model saved to: {SAVE_PATH}")
    print("Next → run: python step2_build_rag.py")

if __name__ == "__main__":
    train()