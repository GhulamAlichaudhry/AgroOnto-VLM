"""
AgroOnto-VLM — IP102 Real Data Collection Script
==================================================
Run this on your local NVIDIA GeForce RTX GPU to get real IP102 per-class
F1 scores and insect-order severity accuracy for the manuscript figures.

Usage:
  python run_ip102_metrics.py              # full train + metrics
  python run_ip102_metrics.py --skip-train # metrics only from saved checkpoint

Output:
  results/metrics_IP102_real.json          # per-class + order-level metrics
  results/checkpoints/agroonto_ip102.pt   # saved model for reuse
"""
import argparse
import os
import sys
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agroonto import config
from agroonto.model import AgroOntoVLM
from agroonto.train import train_one_epoch, evaluate
from agroonto.dataset import build_loaders, get_clip_preprocess
from agroonto.ontology import compute_similarity_matrix

# ── Optimal hyperparameters from sweep ──────────────────────────────────────
LAMBDA_ONTO    = 0.125
LAMBDA_UNIFORM = config.LAMBDA_UNIFORM   # 0.10
LAMBDA_CE      = 1.0 - LAMBDA_UNIFORM - LAMBDA_ONTO   # 0.775
LAMBDA_PROTO   = config.LAMBDA_PROTO     # 0.0
EPOCHS         = 30
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = os.path.join(config.BASE_DIR, "results", "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "agroonto_ip102.pt")
OUTPUT_PATH    = os.path.join(config.BASE_DIR, "results", "metrics_IP102_real.json")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Insect order keywords (best-effort matching on class names) ─────────────
ORDER_KEYWORDS = {
    "Lepidoptera": [
        "moth", "butterfly", "cutworm", "armyworm", "earworm", "bollworm",
        "leafroller", "looper", "webworm", "skipper", "noctuid", "pyralid",
        "tortrix", "codling moth", "diamondback", "spodoptera", "helicoverpa",
        "pieris", "plutella", "manduca", "agrotis", "prodenia", "mythimna",
        "chilo", "diatraea", "ostrinia", "sesamia", "pectinophora",
        "cnaphalocrocis", "crocidolomia", "leucoptera", "phyllocnistis",
        "tuta", "achaea", "anomis", "earias", "heliothis",
    ],
    "Coleoptera": [
        "beetle", "weevil", "chafer", "longhorn", "flea beetle", "leaf beetle",
        "bark beetle", "click beetle", "boll weevil", "colorado potato",
        "asian longhorned", "emerald ash", "western corn rootworm",
        "spotted cucumber", "striped cucumber", "curculio", "diabrotica",
        "anthonomus", "callosobruchus", "sitophilus", "tribolium",
        "leptinotarsa", "phyllotreta", "altica", "plagiodera",
        "chrysomelid", "cerambycid", "scolytid",
    ],
    "Hemiptera": [
        "aphid", "planthopper", "leafhopper", "whitefly", "scale insect",
        "mealybug", "psyllid", "stink bug", "lace bug", "chinch bug",
        "spittlebug", "brown planthopper", "white backed planthopper",
        "green peach aphid", "cotton aphid", "wooly apple aphid",
        "greenhouse whitefly", "silverleaf whitefly", "bemisia",
        "aphis", "myzus", "nilaparvata", "sogatella", "empoasca",
        "nephotettix", "lygus", "halyomorpha", "nezara",
    ],
    "Thysanoptera": [
        "thrips", "western flower thrips", "onion thrips", "frankliniella",
        "thrip", "thysanoptera",
    ],
    "Orthoptera": [
        "locust", "grasshopper", "cricket", "katydid", "termite",
        "migratory locust", "meadow grasshopper", "acridid",
    ],
    "Diptera": [
        "fly", "midge", "gnat", "hessian fly", "melon fly", "fruit fly",
        "leafminer", "agromyzid", "carrot fly", "cabbage fly",
        "wheat midge", "sorghum midge", "phorid", "drosophila",
    ],
    "Hymenoptera": [
        "bee", "wasp", "ant", "hornet", "sawfly", "leafcutter bee",
        "asian giant hornet", "varroa", "bees",
    ],
    "Acari": [
        "mite", "spider mite", "red spider mite", "rust mite", "gall mite",
        "eriophyid", "tetranychid", "panonychus", "tetranychus",
    ],
}

# Orders we want in the figure (matching the estimated figure's order list)
FIGURE_ORDERS = ["Lepidoptera", "Coleoptera", "Hemiptera",
                 "Thysanoptera", "Orthoptera", "Diptera", "Hymenoptera", "Acari"]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def match_order(class_name):
    """Return the insect order for a class name using keyword matching."""
    name_lower = class_name.lower()
    for order, keywords in ORDER_KEYWORDS.items():
        for kw in keywords:
            if kw in name_lower:
                return order
    return "Other"


def compute_per_class_metrics(model, val_loader, class_names, device):
    """Return per-class precision/recall/F1, plus raw prediction arrays."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast():
                logits_cls, _, _ = model(images)
            preds = logits_cls.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    num_classes = len(class_names)

    per_class = {}
    for c in range(num_classes):
        tp = int(((all_preds == c) & (all_labels == c)).sum())
        fp = int(((all_preds == c) & (all_labels != c)).sum())
        fn = int(((all_preds != c) & (all_labels == c)).sum())

        prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[class_names[c]] = {
            "precision": round(float(prec), 4),
            "recall":    round(float(rec),  4),
            "f1":        round(float(f1),   4),
            "support":   int((all_labels == c).sum()),
            "order":     match_order(class_names[c]),
        }

    macro_f1   = float(np.mean([v["f1"]  for v in per_class.values()]))
    macro_prec = float(np.mean([v["precision"] for v in per_class.values()]))
    macro_rec  = float(np.mean([v["recall"]    for v in per_class.values()]))
    accuracy   = float((all_preds == all_labels).mean() * 100)

    return {
        "accuracy":        round(accuracy, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall":    round(macro_rec, 4),
        "macro_f1":        round(macro_f1, 4),
        "per_class":       per_class,
    }, all_preds, all_labels


def compute_order_metrics(per_class_metrics):
    """Aggregate per-class results into insect-order level accuracy/F1."""
    order_data = {}
    for cls_name, m in per_class_metrics["per_class"].items():
        order = m["order"]
        if order not in order_data:
            order_data[order] = {"f1_sum": 0.0, "prec_sum": 0.0, "rec_sum": 0.0,
                                  "count": 0, "classes": []}
        order_data[order]["f1_sum"]   += m["f1"]
        order_data[order]["prec_sum"] += m["precision"]
        order_data[order]["rec_sum"]  += m["recall"]
        order_data[order]["count"]    += 1
        order_data[order]["classes"].append(cls_name)

    order_summary = {}
    for order, d in sorted(order_data.items(), key=lambda x: -x[1]["count"]):
        n = d["count"]
        order_summary[order] = {
            "mean_f1":        round(d["f1_sum"]   / n, 4),
            "mean_precision": round(d["prec_sum"] / n, 4),
            "mean_recall":    round(d["rec_sum"]  / n, 4),
            "num_classes":    n,
            "classes":        d["classes"],
        }
    return order_summary


def train_ip102(preprocess):
    """Train AgroOnto-VLM on IP102 with optimal hyperparameters."""
    set_seed(42)

    print(f"\n{'='*65}")
    print(f"  IP102 Training — AgroOnto-VLM (Optimal HP)")
    print(f"  lambda_onto={LAMBDA_ONTO}  lambda_ce={LAMBDA_CE:.3f}  lambda_uniform={LAMBDA_UNIFORM}")
    print(f"  Epochs={EPOCHS}  Device={DEVICE}")
    if DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name}  |  VRAM: {props.total_memory/1024**3:.1f} GB")
    print(f"{'='*65}\n")

    train_loader, val_loader, class_names, num_classes = build_loaders("IP102", preprocess)
    print(f"  Classes: {num_classes}  |  Train: {len(train_loader.dataset)}  |"
          f"  Val: {len(val_loader.dataset)}")

    onto_sim = compute_similarity_matrix("IP102", class_names, num_classes).to(DEVICE)
    model    = AgroOntoVLM(num_classes, class_names, device=DEVICE).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.get_param_groups(config.LR_LORA, config.LR_HEAD,
                               config.LR_DELTA, config.LR_LOGIT_SCALE),
        weight_decay=config.WEIGHT_DECAY)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    best_epoch   = 0
    history      = {"train_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Acc':>8}  {'Best':>8}  {'ETA':>8}")
    print(f"  {'─'*60}")

    epoch_times = []
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, onto_sim,
            num_classes, LAMBDA_CE, LAMBDA_UNIFORM, LAMBDA_ONTO,
            LAMBDA_PROTO, DEVICE)
        val_acc = evaluate(model, val_loader, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0
        epoch_times.append(elapsed)

        history["train_loss"].append(round(loss, 4))
        history["train_acc"].append(round(train_acc, 2))
        history["val_acc"].append(round(val_acc, 2))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_names": class_names,
                "num_classes": num_classes,
                "lambda_onto": LAMBDA_ONTO,
            }, CHECKPOINT_PATH)
            marker = " ← best"
        else:
            marker = ""

        remaining = np.mean(epoch_times) * (EPOCHS - epoch)
        eta = f"{int(remaining//60)}m{int(remaining%60):02d}s"
        print(f"  {epoch:>5}  {loss:>10.4f}  {train_acc:>8.2f}%  {val_acc:>7.2f}%"
              f"  {best_val_acc:>7.2f}%  {eta:>8}{marker}")

    print(f"\n  Training complete.")
    print(f"  Best val accuracy: {best_val_acc:.4f}% at epoch {best_epoch}")
    print(f"  Checkpoint saved: {CHECKPOINT_PATH}")
    return model, val_loader, class_names, num_classes, best_val_acc, history


def main():
    parser = argparse.ArgumentParser(description="IP102 real data collection")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training; load checkpoint for inference only")
    args = parser.parse_args()

    preprocess = get_clip_preprocess()

    if args.skip_train:
        if not os.path.exists(CHECKPOINT_PATH):
            print(f"ERROR: No checkpoint found at {CHECKPOINT_PATH}")
            print("       Run without --skip-train to train first.")
            sys.exit(1)

        print(f"\n  Loading checkpoint: {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        class_names  = ckpt["class_names"]
        num_classes  = ckpt["num_classes"]
        _, val_loader, _, _ = build_loaders("IP102", preprocess)

        model = AgroOntoVLM(num_classes, class_names, device=DEVICE).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Checkpoint epoch={ckpt['epoch']}  val_acc={ckpt['val_acc']:.4f}%")
        history = {}
        best_val_acc = ckpt["val_acc"]
    else:
        model, val_loader, class_names, num_classes, best_val_acc, history = \
            train_ip102(preprocess)

    # ── Compute per-class metrics ─────────────────────────────────────────
    print(f"\n  Computing per-class metrics on validation set…")
    metrics, all_preds, all_labels = compute_per_class_metrics(
        model, val_loader, class_names, DEVICE)
    order_summary = compute_order_metrics(metrics)

    # ── Print per-class table ─────────────────────────────────────────────
    print(f"\n  {'Class':<45} {'Order':<15} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>5}")
    print(f"  {'─'*90}")
    for cls_name, m in sorted(metrics["per_class"].items(),
                               key=lambda x: -x[1]["f1"]):
        short = cls_name[:43]
        print(f"  {short:<45} {m['order']:<15} {m['precision']:.4f} "
              f"{m['recall']:.4f} {m['f1']:.4f} {m['support']:>5}")

    print(f"\n  Overall Accuracy : {metrics['accuracy']:.4f}%")
    print(f"  Macro Precision  : {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall     : {metrics['macro_recall']:.4f}")
    print(f"  Macro F1         : {metrics['macro_f1']:.4f}")

    # ── Print order-level table ───────────────────────────────────────────
    print(f"\n  {'Order':<18} {'Classes':>7} {'Mean F1':>8} {'Mean Prec':>10} {'Mean Rec':>9}")
    print(f"  {'─'*58}")
    for order, od in order_summary.items():
        print(f"  {order:<18} {od['num_classes']:>7} {od['mean_f1']:>8.4f} "
              f"{od['mean_precision']:>10.4f} {od['mean_recall']:>9.4f}")

    unmatched = order_summary.get("Other", {}).get("num_classes", 0)
    if unmatched:
        print(f"\n  NOTE: {unmatched} classes not matched to any order (labelled 'Other').")
        print(f"  Classes: {', '.join(order_summary['Other']['classes'][:10])}"
              + ("…" if unmatched > 10 else ""))

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "dataset":      "IP102",
        "lambda_onto":  LAMBDA_ONTO,
        "lambda_ce":    LAMBDA_CE,
        "best_val_acc": best_val_acc,
        "accuracy":     metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall":    metrics["macro_recall"],
        "macro_f1":        metrics["macro_f1"],
        "per_class":    metrics["per_class"],
        "order_summary": order_summary,
        "history":      history,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  Done!  Results saved to:")
    print(f"  {OUTPUT_PATH}")
    print(f"{'='*65}")
    print(f"\n  NEXT STEP: Share '{OUTPUT_PATH}' so the manuscript figures")
    print(f"  can be updated with your real IP102 simulation results.")


if __name__ == "__main__":
    main()
