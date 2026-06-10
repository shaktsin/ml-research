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

def clip_subject_gradient(grads: dict[str, torch.Tensor], clip_norm: float) -> dict[str, torch.Tensor]:
    """Clip the full subject gradient to L2 norm <= clip_norm."""
    total_norm = torch.sqrt(
        sum(grad.pow(2).sum() for grad in grads.values())
    )
    scale = torch.clamp(clip_norm / (total_norm + 1e-8), max=1.0)
    return {name: grad * scale for name, grad in grads.items()}


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
      4. Sum clipped subject gradients
      5. Add one noise tensor to the summed gradient
      6. Average across subjects → apply update
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
            accumulated = {}  # param_name -> accumulated clipped gradient
            noise_multipliers = []

            for sid, grad_list in per_sample_grads.items():
                n_samples = len(grad_list)

                # Average gradients across this subject's samples (eq. 1)
                subject_grad = {}
                for name in grad_list[0]:
                    subject_grad[name] = torch.stack(
                        [g[name] for g in grad_list]
                    ).mean(dim=0)

                # Clip the complete subject gradient, not each tensor independently.
                subject_grad = clip_subject_gradient(subject_grad, clip_norm)

                sigma = adaptive_noise_scale(base_sigma, n_samples) if adaptive else base_sigma
                noise_multipliers.append(sigma)

                for name in subject_grad:
                    if name not in accumulated:
                        accumulated[name] = subject_grad[name]
                    else:
                        accumulated[name] += subject_grad[name]

            # Add one Gaussian noise tensor to the summed gradient, then average.
            # For adaptive mode, use the RMS multiplier across subjects in the batch.
            noise_multiplier = math.sqrt(
                sum(sigma ** 2 for sigma in noise_multipliers) / len(noise_multipliers)
            )
            for name, param in model.named_parameters():
                if name in accumulated:
                    noise = torch.randn_like(accumulated[name]) * noise_multiplier * clip_norm
                    param.grad = (accumulated[name] + noise) / n_subjects

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
