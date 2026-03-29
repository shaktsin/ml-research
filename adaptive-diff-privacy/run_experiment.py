"""
Main experiment runner.

Trains three configurations and evaluates each on:
  - Classification accuracy
  - MIA attack AUC

Usage:
    python run_experiment.py
"""

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AdamW

from data import AGNewsDataset, subject_contribution_stats
from model import get_model, get_tokenizer
from trainer import train_baseline, train_subject_dp, evaluate
from mia import run_mia

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
LR = 2e-5
BATCH_SIZE = 16
EPOCHS = 3
CLIP_NORM = 1.0
BASE_SIGMA = 1.0
N_SUBJECTS = 500
# Limit samples for quick runs — set to None to use full dataset
MAX_TRAIN = 2000
MAX_TEST = 500


def make_loaders(tokenizer):
    print("Loading AG News dataset...")
    train_ds = AGNewsDataset("train", tokenizer, n_subjects=N_SUBJECTS)
    test_ds = AGNewsDataset("test", tokenizer, n_subjects=N_SUBJECTS)

    if MAX_TRAIN:
        train_ds = Subset(train_ds, range(MAX_TRAIN))
    if MAX_TEST:
        test_ds = Subset(test_ds, range(MAX_TEST))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    # Small fixed subset for MIA evaluation (balanced member/non-member)
    mia_train_loader = DataLoader(Subset(train_ds, range(200)), batch_size=BATCH_SIZE)
    mia_test_loader = DataLoader(Subset(test_ds, range(200)), batch_size=BATCH_SIZE)
    return train_loader, test_loader, mia_train_loader, mia_test_loader


def run_config(name: str, train_fn, train_loader, test_loader,
               mia_train_loader, mia_test_loader, tokenizer):
    print(f"\n{'='*60}")
    print(f"  Config: {name}")
    print(f"{'='*60}")

    model = get_model(num_labels=4).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)

    train_fn(model, train_loader, optimizer, DEVICE, EPOCHS)

    acc = evaluate(model, test_loader, DEVICE)
    mia_auc = run_mia(model, mia_train_loader, mia_test_loader, DEVICE)

    print(f"\n  Results for [{name}]")
    print(f"    Accuracy : {acc:.4f}")
    print(f"    MIA AUC  : {mia_auc:.4f}  (0.5=best privacy, 1.0=worst)")

    return {"config": name, "accuracy": acc, "mia_auc": mia_auc}


def main():
    print(f"Device: {DEVICE}")
    tokenizer = get_tokenizer()
    train_loader, test_loader, mia_train, mia_test = make_loaders(tokenizer)

    results = []

    # 1. Baseline — no DP
    results.append(run_config(
        name="Baseline (no DP)",
        train_fn=train_baseline,
        train_loader=train_loader,
        test_loader=test_loader,
        mia_train_loader=mia_train,
        mia_test_loader=mia_test,
        tokenizer=tokenizer,
    ))

    # 2. Subject-level DP — uniform noise
    results.append(run_config(
        name="Subject-DP (uniform noise)",
        train_fn=lambda m, tl, opt, dev, ep: train_subject_dp(
            m, tl, opt, dev, ep,
            clip_norm=CLIP_NORM, base_sigma=BASE_SIGMA, adaptive=False
        ),
        train_loader=train_loader,
        test_loader=test_loader,
        mia_train_loader=mia_train,
        mia_test_loader=mia_test,
        tokenizer=tokenizer,
    ))

    # 3. Adaptive subject-level DP — proposed method
    results.append(run_config(
        name="Adaptive Subject-DP (proposed)",
        train_fn=lambda m, tl, opt, dev, ep: train_subject_dp(
            m, tl, opt, dev, ep,
            clip_norm=CLIP_NORM, base_sigma=BASE_SIGMA, adaptive=True
        ),
        train_loader=train_loader,
        test_loader=test_loader,
        mia_train_loader=mia_train,
        mia_test_loader=mia_test,
        tokenizer=tokenizer,
    ))

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Config':<35} {'Accuracy':>10} {'MIA AUC':>10}")
    print(f"  {'-'*55}")
    for r in results:
        print(f"  {r['config']:<35} {r['accuracy']:>10.4f} {r['mia_auc']:>10.4f}")
    print()
    print("  MIA AUC interpretation:")
    print("    ~0.50 → model leaks nothing (ideal privacy)")
    print("    ~1.00 → model leaks membership perfectly (no privacy)")


if __name__ == "__main__":
    main()
