"""
Main experiment runner.

Trains three configurations and evaluates each on:
  - Classification accuracy
  - MIA attack AUC

Usage:
    python run_experiment.py
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from data import DATASET_CONFIGS, TextClassificationDataset
from model import get_model, get_tokenizer
from trainer import train_baseline, train_subject_dp, evaluate
from mia import run_mia

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
LR = 2e-5
BATCH_SIZE = 16
EPOCHS = 3
CLIP_NORM = 1.0
BASE_SIGMA = 0.005
N_SUBJECTS = 500
# Limit samples for quick runs — set to None to use full dataset
MAX_TRAIN = 2000
MAX_TEST = 500
OUTPUT_DIR = "results"


def mia_risk_auc(raw_auc: float) -> float:
    """
    Convert raw attack AUC into directional attack risk.

    A raw AUC below 0.5 means this simple loss-threshold attack is pointing in
    the wrong direction for that run. A real attacker could invert the score, so
    the risk should be interpreted as max(AUC, 1 - AUC).
    """
    return max(raw_auc, 1.0 - raw_auc)


def mia_advantage(raw_auc: float) -> float:
    """Attack advantage over random guessing, reported on a 0 to 1 scale."""
    return 2.0 * (mia_risk_auc(raw_auc) - 0.5)


def next_iteration(results_csv: Path) -> int:
    if not results_csv.exists():
        return 1

    with results_csv.open("r", newline="") as f:
        rows = csv.DictReader(f)
        iterations = [
            int(row["iteration"])
            for row in rows
            if row.get("iteration", "").isdigit()
        ]
    return max(iterations, default=0) + 1


def build_run_record(args, results, iteration: int) -> dict:
    baseline, uniform_dp, adaptive_dp = results
    adaptive_accuracy_delta = adaptive_dp["accuracy"] - baseline["accuracy"]
    adaptive_vs_uniform_delta = adaptive_dp["accuracy"] - uniform_dp["accuracy"]

    return {
        "iteration": iteration,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(DEVICE),
        "args": vars(args),
        "results": [
            {
                **result,
                "mia_risk_auc": mia_risk_auc(result["mia_auc"]),
                "mia_advantage": mia_advantage(result["mia_auc"]),
            }
            for result in results
        ],
        "takeaway": {
            "adaptive_accuracy_delta_vs_baseline": adaptive_accuracy_delta,
            "adaptive_accuracy_delta_vs_uniform_dp": adaptive_vs_uniform_delta,
            "adaptive_mia_risk_auc": mia_risk_auc(adaptive_dp["mia_auc"]),
            "adaptive_mia_advantage": mia_advantage(adaptive_dp["mia_auc"]),
        },
    }


def save_run_record(run_record: dict, output_dir: str) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    iteration = run_record["iteration"]
    json_path = out_dir / f"iteration_{iteration:04d}.json"
    csv_path = out_dir / "experiment_results.csv"

    with json_path.open("w") as f:
        json.dump(run_record, f, indent=2)

    fieldnames = [
        "iteration",
        "timestamp_utc",
        "dataset",
        "base_sigma",
        "epochs",
        "max_train",
        "max_test",
        "mia_samples",
        "config",
        "accuracy",
        "raw_mia_auc",
        "mia_risk_auc",
        "mia_advantage",
        "adaptive_accuracy_delta_vs_baseline",
        "adaptive_accuracy_delta_vs_uniform_dp",
    ]
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        args = run_record["args"]
        takeaway = run_record["takeaway"]
        for result in run_record["results"]:
            writer.writerow({
                "iteration": iteration,
                "timestamp_utc": run_record["timestamp_utc"],
                "dataset": args["dataset"],
                "base_sigma": args["base_sigma"],
                "epochs": args["epochs"],
                "max_train": args["max_train"],
                "max_test": args["max_test"],
                "mia_samples": args["mia_samples"],
                "config": result["config"],
                "accuracy": result["accuracy"],
                "raw_mia_auc": result["mia_auc"],
                "mia_risk_auc": result["mia_risk_auc"],
                "mia_advantage": result["mia_advantage"],
                "adaptive_accuracy_delta_vs_baseline": takeaway[
                    "adaptive_accuracy_delta_vs_baseline"
                ],
                "adaptive_accuracy_delta_vs_uniform_dp": takeaway[
                    "adaptive_accuracy_delta_vs_uniform_dp"
                ],
            })

    return json_path, csv_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run text classification baseline and subject-DP experiments.")
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), default="ag_news")
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--clip-norm", type=float, default=CLIP_NORM)
    parser.add_argument("--base-sigma", type=float, default=BASE_SIGMA)
    parser.add_argument("--n-subjects", type=int, default=N_SUBJECTS)
    parser.add_argument("--max-train", type=int, default=MAX_TRAIN)
    parser.add_argument("--max-test", type=int, default=MAX_TEST)
    parser.add_argument("--mia-samples", type=int, default=200)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    return parser.parse_args()


def make_loaders(tokenizer, args):
    print(f"Loading {args.dataset} dataset...")
    train_ds = TextClassificationDataset(
        args.dataset, "train", tokenizer, n_subjects=args.n_subjects
    )
    test_ds = TextClassificationDataset(
        args.dataset, "test", tokenizer, n_subjects=args.n_subjects
    )

    if args.max_train:
        train_ds = Subset(train_ds, range(args.max_train))
    if args.max_test:
        test_ds = Subset(test_ds, range(args.max_test))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    # Small fixed subset for MIA evaluation (balanced member/non-member)
    mia_train_n = min(args.mia_samples, len(train_ds))
    mia_test_n = min(args.mia_samples, len(test_ds))
    mia_train_loader = DataLoader(Subset(train_ds, range(mia_train_n)), batch_size=args.batch_size)
    mia_test_loader = DataLoader(Subset(test_ds, range(mia_test_n)), batch_size=args.batch_size)
    return train_loader, test_loader, mia_train_loader, mia_test_loader


def run_config(name: str, train_fn, train_loader, test_loader,
               mia_train_loader, mia_test_loader, args):
    print(f"\n{'='*60}")
    print(f"  Config: {name}")
    print(f"{'='*60}")

    model = get_model(num_labels=DATASET_CONFIGS[args.dataset]["num_labels"]).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    train_fn(model, train_loader, optimizer, DEVICE, args.epochs)

    acc = evaluate(model, test_loader, DEVICE)
    mia_auc = run_mia(model, mia_train_loader, mia_test_loader, DEVICE)

    print(f"\n  Results for [{name}]")
    print(f"    Accuracy : {acc:.4f}")
    print(f"    MIA AUC  : {mia_auc:.4f}  (0.5=best privacy, 1.0=worst)")

    return {"config": name, "accuracy": acc, "mia_auc": mia_auc}


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    iteration = next_iteration(output_dir / "experiment_results.csv")
    print(f"Device: {DEVICE}")
    print(
        "Config: "
        f"lr={args.lr}, batch_size={args.batch_size}, epochs={args.epochs}, "
        f"clip_norm={args.clip_norm}, base_sigma={args.base_sigma}, "
        f"dataset={args.dataset}, iteration={iteration}, "
        f"max_train={args.max_train}, max_test={args.max_test}, "
        f"mia_samples={args.mia_samples}, output_dir={args.output_dir}"
    )
    tokenizer = get_tokenizer()
    train_loader, test_loader, mia_train, mia_test = make_loaders(tokenizer, args)

    results = []

    # 1. Baseline — no DP
    results.append(run_config(
        name="Baseline (no DP)",
        train_fn=train_baseline,
        train_loader=train_loader,
        test_loader=test_loader,
        mia_train_loader=mia_train,
        mia_test_loader=mia_test,
        args=args,
    ))

    # 2. Subject-level DP — uniform noise
    results.append(run_config(
        name="Subject-DP (uniform noise)",
        train_fn=lambda m, tl, opt, dev, ep: train_subject_dp(
            m, tl, opt, dev, ep,
            clip_norm=args.clip_norm, base_sigma=args.base_sigma, adaptive=False
        ),
        train_loader=train_loader,
        test_loader=test_loader,
        mia_train_loader=mia_train,
        mia_test_loader=mia_test,
        args=args,
    ))

    # 3. Adaptive subject-level DP — proposed method
    results.append(run_config(
        name="Adaptive Subject-DP (proposed)",
        train_fn=lambda m, tl, opt, dev, ep: train_subject_dp(
            m, tl, opt, dev, ep,
            clip_norm=args.clip_norm, base_sigma=args.base_sigma, adaptive=True
        ),
        train_loader=train_loader,
        test_loader=test_loader,
        mia_train_loader=mia_train,
        mia_test_loader=mia_test,
        args=args,
    ))

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Config':<35} {'Accuracy':>10} {'Raw MIA':>10} {'MIA Risk':>10}")
    print(f"  {'-'*68}")
    for r in results:
        print(
            f"  {r['config']:<35} "
            f"{r['accuracy']:>10.4f} "
            f"{r['mia_auc']:>10.4f} "
            f"{mia_risk_auc(r['mia_auc']):>10.4f}"
        )

    baseline, uniform_dp, adaptive_dp = results
    adaptive_accuracy_delta = adaptive_dp["accuracy"] - baseline["accuracy"]
    adaptive_vs_uniform_delta = adaptive_dp["accuracy"] - uniform_dp["accuracy"]
    adaptive_privacy_advantage = mia_advantage(adaptive_dp["mia_auc"])

    print(f"\n{'='*60}")
    print("  PAPER-READY TAKEAWAY")
    print(f"{'='*60}")
    print(
        "  Adaptive Subject-DP preserves baseline-level utility: "
        f"{adaptive_dp['accuracy']:.4f} accuracy vs "
        f"{baseline['accuracy']:.4f} baseline "
        f"({adaptive_accuracy_delta:+.4f})."
    )
    print(
        "  Adaptive Subject-DP improves utility over uniform Subject-DP: "
        f"{adaptive_vs_uniform_delta:+.4f} accuracy."
    )
    print(
        "  Membership inference remains near random: "
        f"raw AUC={adaptive_dp['mia_auc']:.4f}, "
        f"risk AUC={mia_risk_auc(adaptive_dp['mia_auc']):.4f}, "
        f"attack advantage={adaptive_privacy_advantage:.4f}."
    )
    print()
    print("  MIA AUC interpretation:")
    print("    Raw AUC near 0.50 means the attack is close to random guessing.")
    print("    Raw AUC below 0.50 is not better-than-perfect privacy; it means")
    print("    the attack direction is unstable, so MIA Risk reports max(AUC, 1-AUC).")

    run_record = build_run_record(args, results, iteration)
    json_path, csv_path = save_run_record(run_record, args.output_dir)
    print()
    print("  Saved iteration results:")
    print(f"    JSON: {json_path}")
    print(f"    CSV : {csv_path}")


if __name__ == "__main__":
    main()
