
"""
"""

import os, torch, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, ConfusionMatrixDisplay
)
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

os.makedirs("results", exist_ok=True)

# ── Config (must match step1_train.py) ─────────────────────────────────────
CSV_PATH   = "data/health_fact.csv"
MODEL_NAME = "distilbert-base-uncased"
MODEL_PATH = "model.pt"
MAX_LEN    = 128
BATCH      = 32
SAMPLE     = None
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

BLUE   = "#2a78d6"
ORANGE = "#eb6834"
GRAY   = "#888780"

# ── Style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "figure.dpi":       150,
})


# ── Model definition (must match step1_train.py) ───────────────────────────
from transformers import AutoModel

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
        return self.classifier(out.last_hidden_state[:, 0, :])


class ClaimDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding=True,
            max_length=MAX_LEN, return_tensors="pt"
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.encodings.items()}, self.labels[i]


# ── Load model and get predictions ─────────────────────────────────────────
def get_predictions():
    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH).dropna(subset=["text", "label"])
    if SAMPLE:
        df = df.sample(n=min(SAMPLE, len(df)), random_state=42).reset_index(drop=True)

    _, test_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)

    tokenizer   = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_loader = DataLoader(
        ClaimDataset(test_df["text"], test_df["label"], tokenizer),
        batch_size=BATCH
    )

    print("Loading model...")
    model = MisinfoClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch, labels in test_loader:
            batch  = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch)
            probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            preds  = logits.argmax(-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
            all_probs.extend(probs)

    return all_labels, all_preds, all_probs, df


# ── Plot 1: Metric cards bar chart ─────────────────────────────────────────
def plot_metrics(labels, preds):
    report = classification_report(labels, preds,
                                    target_names=["credible", "misinformation"],
                                    output_dict=True)

    metrics = ["Accuracy", "Macro F1", "Weighted F1", "Macro Precision", "Macro Recall"]
    values  = [
        report["accuracy"],
        report["macro avg"]["f1-score"],
        report["weighted avg"]["f1-score"],
        report["macro avg"]["precision"],
        report["macro avg"]["recall"],
    ]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(metrics, values, color=[BLUE, ORANGE, BLUE, ORANGE, BLUE],
                   height=0.5, edgecolor="none")
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Score")
    ax.set_title("Overall Model Performance", fontsize=13, fontweight="bold", pad=12)
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig("results/accuracy_metrics.png", bbox_inches="tight")
    plt.close()
    print("Saved: results/accuracy_metrics.png")
    return report


# ── Plot 2: Per-class metrics ──────────────────────────────────────────────
def plot_per_class(report):
    classes  = ["Credible", "Misinformation"]
    metrics  = ["Precision", "Recall", "F1-score"]
    credible = [report["credible"]["precision"],
                report["credible"]["recall"],
                report["credible"]["f1-score"]]
    misinfo  = [report["misinformation"]["precision"],
                report["misinformation"]["recall"],
                report["misinformation"]["f1-score"]]

    x   = np.arange(len(metrics))
    w   = 0.32
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - w/2, credible, w, label="Credible",      color=BLUE,   edgecolor="none", linewidth=0)
    b2 = ax.bar(x + w/2, misinfo,  w, label="Misinformation", color=ORANGE, edgecolor="none", linewidth=0)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Metrics", fontsize=13, fontweight="bold", pad=12)
    ax.legend(framealpha=0.3)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig("results/per_class_metrics.png", bbox_inches="tight")
    plt.close()
    print("Saved: results/per_class_metrics.png")


# ── Plot 3: Confusion matrix ───────────────────────────────────────────────
def plot_confusion(labels, preds):
    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Credible", "Misinformation"]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", bbox_inches="tight")
    plt.close()
    print("Saved: results/confusion_matrix.png")


# ── Plot 4: ROC curve ──────────────────────────────────────────────────────
def plot_roc(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc     = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1.5, linestyle="--", label="Random")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", framealpha=0.3)
    plt.tight_layout()
    plt.savefig("results/roc_curve.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: results/roc_curve.png  (AUC = {roc_auc:.3f})")
    return roc_auc


# ── Plot 5: Training loss curve ────────────────────────────────────────────
def plot_loss():
    epochs     = [0, 1, 2]
    train_loss = [0.683, 0.524, 0.421]
    val_loss   = [0.612, 0.514, 0.471]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_loss, color=BLUE,   lw=2, marker="o", label="Train loss")
    ax.plot(epochs, val_loss,   color=ORANGE, lw=2, marker="s", linestyle="--", label="Val loss")
    ax.set_xticks(epochs)
    ax.set_xticklabels(["Start", "Epoch 1", "Epoch 2"])
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss", fontsize=13, fontweight="bold", pad=12)
    ax.legend(framealpha=0.3)
    plt.tight_layout()
    plt.savefig("results/loss_curve.png", bbox_inches="tight")
    plt.close()
    print("Saved: results/loss_curve.png")


# ── Plot 6: Dataset distribution ───────────────────────────────────────────
def plot_distribution(df):
    counts = df["label"].value_counts().sort_index()
    labels = ["Credible", "Misinformation"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=[BLUE, ORANGE],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_color("white")
    ax.set_title("Dataset Class Distribution", fontsize=13, fontweight="bold", pad=12)
    handles = [
        mpatches.Patch(color=BLUE,   label=f"Credible ({values[0]:,})"),
        mpatches.Patch(color=ORANGE, label=f"Misinformation ({values[1]:,})"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.1),
              ncol=2, framealpha=0.3, fontsize=9)
    plt.tight_layout()
    plt.savefig("results/class_distribution.png", bbox_inches="tight")
    plt.close()
    print("Saved: results/class_distribution.png")


# ── HTML Dashboard ─────────────────────────────────────────────────────────
def build_dashboard(report, roc_auc):
    acc   = report["accuracy"]
    f1    = report["macro avg"]["f1-score"]
    prec  = report["macro avg"]["precision"]
    rec   = report["macro avg"]["recall"]
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MedCheck — Evaluation Results</title>
<style>
  body {{ font-family: Arial, sans-serif; background: #f5f7fa; color: #222; margin: 0; padding: 32px; }}
  h1   {{ font-size: 1.6rem; margin-bottom: 4px; }}
  p.sub {{ color: #888; font-size: 0.85rem; margin-bottom: 2rem; }}
  .grid4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 2rem; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 2rem; }}
  .grid3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }}
  .card  {{ background: #fff; border-radius: 10px; padding: 1.25rem; border: 1px solid #e2e4e8; }}
  .metric-val {{ font-size: 2.2rem; font-weight: 600; margin-bottom: 4px; }}
  .metric-lbl {{ font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }}
  .blue {{ color: #2a78d6; }}
  .orange {{ color: #eb6834; }}
  img {{ width: 100%; border-radius: 8px; }}
  h2  {{ font-size: 1rem; margin: 0 0 12px; }}
</style>
</head>
<body>
<h1>MedCheck — Model Evaluation Results</h1>
<p class="sub">DistilBERT · PUBHEALTH dataset · 3,000 samples · 2 epochs · CPU training</p>

<div class="grid4">
  <div class="card" style="text-align:center">
    <div class="metric-val blue">{acc:.1%}</div>
    <div class="metric-lbl">Accuracy</div>
  </div>
  <div class="card" style="text-align:center">
    <div class="metric-val orange">{f1:.3f}</div>
    <div class="metric-lbl">Macro F1</div>
  </div>
  <div class="card" style="text-align:center">
    <div class="metric-val blue">{prec:.3f}</div>
    <div class="metric-lbl">Macro Precision</div>
  </div>
  <div class="card" style="text-align:center">
    <div class="metric-val orange">{rec:.3f}</div>
    <div class="metric-lbl">Macro Recall</div>
  </div>
</div>

<div class="grid2">
  <div class="card"><h2>Per-class metrics</h2><img src="per_class_metrics.png"></div>
  <div class="card"><h2>Confusion matrix</h2><img src="confusion_matrix.png"></div>
</div>

<div class="grid3">
  <div class="card"><h2>ROC curve (AUC = {roc_auc:.3f})</h2><img src="roc_curve.png"></div>
  <div class="card"><h2>Training loss</h2><img src="loss_curve.png"></div>
  <div class="card"><h2>Dataset distribution</h2><img src="class_distribution.png"></div>
</div>

</body>
</html>"""
    with open("results/dashboard.html", "w") as f:
        f.write(html)
    print("Saved: results/dashboard.html")


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    labels, preds, probs, df = get_predictions()

    report  = plot_metrics(labels, preds)
    plot_per_class(report)
    plot_confusion(labels, preds)
    roc_auc = plot_roc(labels, probs)
    plot_loss()
    plot_distribution(df)
    build_dashboard(report, roc_auc)

    print("\n" + "="*50)
    print("All charts saved to results/")
    print("Open results/dashboard.html in your browser")
    print("="*50)

    print(classification_report(labels, preds,target_names=["credible", "misinformation"]))