"""
Three training modes:
  1. baseline     — standard fine-tuning, no DP
  2. subject_dp   — subject-level DP with uniform noise
  3. adaptive_dp  — subject-level DP with adaptive noise (proposed method)

Subject-level gradient aggregation is implemented manually on top of
standard PyTorch, since Opacus operates at record level by default.
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Gradient utilities
# ---------------------------------------------------------------------------

def clip_gradient(grad: torch.Tensor, clip_norm: float) -> torch.Tensor:
    """Clip a gradient tensor to L2 norm <= clip_norm (equation 2 in paper)."""
    norm = grad.norm(2)
    if norm > clip_norm:
        grad = grad * (clip_norm / (norm + 1e-8))
    return grad


def adaptive_noise_scale(base_sigma: float, n_samples: int) -> float:
    """Equation 3 in paper: σ_u = σ * log(1 + |D_u|)"""
    return base_sigma * math.log(1 + n_samples)


# ---------------------------------------------------------------------------
# Core training functions
# ---------------------------------------------------------------------------

def train_baseline(model, train_loader, optimizer, device, epochs):
    """Standard fine-tuning without any DP."""
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Baseline epoch {epoch+1}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()

        print(f"  Epoch {epoch+1} loss: {total_loss / len(train_loader):.4f}")


def train_subject_dp(model, train_loader, optimizer, device, epochs,
                     clip_norm: float, base_sigma: float, adaptive: bool):
    """
    Subject-level DP training.

    Steps per batch:
      1. Compute per-sample gradients
      2. Group by subject_id → average within subject
      3. Clip subject gradient
      4. Add noise (uniform or adaptive depending on `adaptive` flag)
      5. Average across subjects → apply update
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    for epoch in range(epochs):
        total_loss = 0.0
        mode = "Adaptive-DP" if adaptive else "Subject-DP"

        for batch in tqdm(train_loader, desc=f"{mode} epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            subject_ids = batch["subject_id"]  # list of ints, stays on CPU

            batch_size = input_ids.size(0)

            # --- Step 1: per-sample gradients ---
            per_sample_grads = defaultdict(list)  # subject_id -> list of grad dicts

            for i in range(batch_size):
                optimizer.zero_grad()
                out = model(
                    input_ids=input_ids[i].unsqueeze(0),
                    attention_mask=attention_mask[i].unsqueeze(0),
                )
                logits = out.logits
                loss = loss_fn(logits, labels[i].unsqueeze(0)).mean()
                loss.backward()

                sid = int(subject_ids[i])
                grads = {
                    name: param.grad.detach().clone()
                    for name, param in model.named_parameters()
                    if param.grad is not None
                }
                per_sample_grads[sid].append(grads)
                total_loss += loss.item()

            # --- Steps 2-5: subject aggregation, clip, noise, average ---
            optimizer.zero_grad()

            n_subjects = len(per_sample_grads)
            accumulated = {}  # param_name -> accumulated noised gradient

            for sid, grad_list in per_sample_grads.items():
                n_samples = len(grad_list)

                # Average gradients across this subject's samples (eq. 1)
                subject_grad = {}
                for name in grad_list[0]:
                    subject_grad[name] = torch.stack(
                        [g[name] for g in grad_list]
                    ).mean(dim=0)

                # Clip subject gradient (eq. 2) — applied per-parameter
                for name in subject_grad:
                    subject_grad[name] = clip_gradient(subject_grad[name], clip_norm)

                # Noise scale (eq. 3)
                sigma = adaptive_noise_scale(base_sigma, n_samples) if adaptive else base_sigma

                # Add noise and accumulate
                for name in subject_grad:
                    noise = torch.randn_like(subject_grad[name]) * sigma * clip_norm
                    noised = subject_grad[name] + noise
                    if name not in accumulated:
                        accumulated[name] = noised
                    else:
                        accumulated[name] += noised

            # Average across subjects (eq. 4) and write into .grad
            for name, param in model.named_parameters():
                if name in accumulated:
                    param.grad = accumulated[name] / n_subjects

            optimizer.step()

        print(f"  Epoch {epoch+1} loss: {total_loss / (len(train_loader) * train_loader.batch_size):.4f}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total
