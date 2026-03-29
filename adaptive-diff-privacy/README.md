# Adaptive Subject-Level Differential Privacy for NLP

An experiment implementing adaptive noise scaling for subject-level differential privacy in NLP training, based on the paper:

> **"Adaptive Noise Scaling for Subject-Level Differential Privacy in NLP Training"**

---

## The Problem

Standard DP-SGD protects each training record equally. But when one user contributes hundreds of records, their fingerprint remains detectable even after noise injection. This project protects at the **subject (person) level** — scaling noise by contribution count.

---

## Approach

Three training configurations are compared on AG News text classification (DistilBERT):

| Config | Privacy | Accuracy | MIA AUC |
|--------|---------|----------|---------|
| Baseline (no DP) | None | ~88% | ~0.85 |
| Subject-DP (uniform noise) | Record-level | ~84% | ~0.65 |
| **Adaptive Subject-DP** (proposed) | **Subject-level** | ~84% | **~0.55** |

MIA AUC of 0.5 = attacker is guessing randomly (ideal privacy).

---

## How It Works

1. **Group gradients by subject** — each person contributes one gradient per batch, regardless of how many records they have.
2. **Clip** the subject gradient to bound influence.
3. **Adaptive noise** — heavier contributors receive proportionally more noise: `σ_u = σ × log(1 + |records_u|)`

---

## Project Structure

```
adaptive-diff-privacy/
├── data.py             # AG News loading + Zipf subject assignment
├── model.py            # DistilBERT 4-class classifier
├── trainer.py          # Baseline, Subject-DP, and Adaptive-DP training loops
├── mia.py              # Loss-threshold membership inference attack
├── run_experiment.py   # Entry point — runs all 3 configs and prints results
└── requirements.txt    # Dependencies
```

---

## Quick Start

```bash
pip install -r requirements.txt
python run_experiment.py
```

---

## Key References

- Abadi et al., *Deep Learning with Differential Privacy*, CCS 2016
- Shokri et al., *Membership Inference Attacks Against Machine Learning Models*, IEEE S&P 2017
- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers*, NAACL 2019
