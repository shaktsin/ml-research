"""
Membership Inference Attack (MIA) — loss threshold attack.

Intuition: models are overconfident (lower loss) on training data.
An attacker uses loss as a signal to guess membership.

Returns attack AUC — closer to 0.5 means better privacy,
closer to 1.0 means the model leaks membership information.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score


def compute_losses(model, loader: DataLoader, device) -> np.ndarray:
    """Compute per-sample cross-entropy loss for a dataset."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            losses = loss_fn(outputs.logits, labels)
            all_losses.extend(losses.cpu().numpy())

    return np.array(all_losses)


def run_mia(model, train_loader: DataLoader, test_loader: DataLoader, device) -> float:
    """
    Loss-threshold membership inference attack.

    Members (train set) tend to have lower loss → attacker uses
    -loss as the membership score, computes AUC.

    Returns:
        auc: float — attack AUC. 0.5 = random (perfect privacy), 1.0 = full leak.
    """
    train_losses = compute_losses(model, train_loader, device)
    test_losses = compute_losses(model, test_loader, device)

    # Labels: 1 = member (train), 0 = non-member (test)
    labels = np.concatenate([
        np.ones(len(train_losses)),
        np.zeros(len(test_losses)),
    ])
    # Score: lower loss → more likely member → use negative loss as score
    scores = np.concatenate([-train_losses, -test_losses])

    auc = roc_auc_score(labels, scores)
    return auc
