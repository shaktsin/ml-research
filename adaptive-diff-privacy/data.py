"""
Data loading and subject assignment for AG News dataset.
Assigns synthetic subject IDs using a Zipf distribution to simulate
real-world heavy contributors.
"""

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset


def assign_subject_ids(n_samples: int, n_subjects: int, seed: int = 42) -> np.ndarray:
    """
    Assign subject IDs using a Zipf distribution so some subjects
    contribute many samples (heavy contributors) and most contribute few.
    """
    rng = np.random.default_rng(seed)
    # Zipf weights: subject 0 gets most samples, subject n_subjects-1 gets least
    weights = 1.0 / np.arange(1, n_subjects + 1)
    weights /= weights.sum()
    return rng.choice(n_subjects, size=n_samples, p=weights)


class AGNewsDataset(Dataset):
    def __init__(self, split: str, tokenizer, max_length: int = 128,
                 n_subjects: int = 500, seed: int = 42):
        raw = load_dataset("ag_news", split=split)
        self.texts = raw["text"]
        self.labels = raw["label"]
        self.subject_ids = assign_subject_ids(len(self.texts), n_subjects, seed)
        self.encodings = tokenizer(
            list(self.texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
            "subject_id": int(self.subject_ids[idx]),
        }


def subject_contribution_stats(subject_ids: np.ndarray) -> dict:
    """Return basic stats about subject contribution distribution."""
    counts = np.bincount(subject_ids)
    return {
        "n_subjects": len(counts),
        "mean_samples": counts.mean(),
        "max_samples": counts.max(),
        "min_samples": counts.min(),
        "top10_subjects": sorted(counts, reverse=True)[:10],
    }
