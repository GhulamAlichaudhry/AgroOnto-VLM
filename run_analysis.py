"""
AgroOnto-VLM — Post-Training Analysis & Severity Study
=======================================================

Run AFTER all simulations (M1-M7 and P1-P5) are complete.

Generates:
  1. Per-class metrics (Precision, Recall, F1-score) for all 3 datasets
  2. Confusion matrices (saved as PNG)
  3. Disease severity analysis (cross-entropy loss per severity level)
  4. Multi-seed evaluation (3 seeds, mean ± std)
  5. Ontology similarity heatmap
  6. Training curves plots
  7. t-SNE feature visualization

Usage:
  python run_analysis.py                    # run all analyses
  python run_analysis.py --analysis severity      # severity only
  python run_analysis.py --analysis metrics        # per-class metrics only
  python run_analysis.py --analysis confusion      # confusion matrices only
  python run_analysis.py --analysis heatmap        # ontology heatmap only
  python run_analysis.py --analysis tsne           # t-SNE visualization only
  python run_analysis.py --analysis multiseed      # 3-seed evaluation only
  python run_analysis.py --analysis curves         # training curves only
"""
import argparse
import os
import sys
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agroonto import config
from agroonto.model import AgroOntoVLM
from agroonto.train import train_one_epoch, evaluate
from agroonto.dataset import build_loaders, get_clip_preprocess
from agroonto.ontology import compute_similarity_matrix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = os.path.join(config.BASE_DIR, "results", "analysis")
os.makedirs(RESULTS_DIR, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ===================================================================
#  1. Per-Class Metrics (Precision, Recall, F1-Score)
# ===================================================================
def compute_per_class_metrics(model, val_loader, class_names, device):
    """Compute precision, recall, F1 per class."""
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

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    num_classes = len(class_names)

    metrics = {}
    for c in range(num_classes):
        tp = ((all_preds == c) & (all_labels == c)).sum()
        fp = ((all_preds == c) & (all_labels != c)).sum()
        fn = ((all_preds != c) & (all_labels == c)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[class_names[c]] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": int((all_labels == c).sum()),
        }

    macro_p = np.mean([m["precision"] for m in metrics.values()])
    macro_r = np.mean([m["recall"] for m in metrics.values()])
    macro_f1 = np.mean([m["f1"] for m in metrics.values()])
    weighted_f1 = sum(m["f1"] * m["support"] for m in metrics.values()) / len(all_labels)
    accuracy = (all_preds == all_labels).mean() * 100

    summary = {
        "accuracy": round(float(accuracy), 2),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": metrics,
    }
    return summary, all_preds, all_labels


def run_metrics_analysis(datasets=None):
    """Generate per-class metrics for each dataset."""
    if datasets is None:
        datasets = ["PlantVillageTomato", "DeepWeeds", "IP102"]

    print(f"\n{'='*70}")
    print(f"  Per-Class Metrics Analysis")
    print(f"{'='*70}")

    preprocess = get_clip_preprocess()

    for ds in datasets:
        print(f"\n  Dataset: {ds}")
        set_seed(42)
        train_loader, val_loader, class_names, num_classes = build_loaders(ds, preprocess)

        onto_sim = compute_similarity_matrix(ds, class_names, num_classes).to(DEVICE)
        model = AgroOntoVLM(num_classes, class_names, device=DEVICE).to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.get_param_groups(config.LR_LORA, config.LR_HEAD,
                                   config.LR_DELTA, config.LR_LOGIT_SCALE),
            weight_decay=config.WEIGHT_DECAY)
        scaler = GradScaler()

        lambda_onto = 0.05
        lambda_ce = 1.0 - config.LAMBDA_UNIFORM - lambda_onto

        print(f"  Training {config.EPOCHS} epochs...")
        for epoch in range(1, config.EPOCHS + 1):
            loss, acc = train_one_epoch(
                model, train_loader, optimizer, scaler, onto_sim,
                num_classes, lambda_ce, config.LAMBDA_UNIFORM, lambda_onto,
                config.LAMBDA_PROTO, DEVICE)
            if epoch % 10 == 0 or epoch == config.EPOCHS:
                val_acc = evaluate(model, val_loader, DEVICE)
                print(f"    Epoch {epoch}/{config.EPOCHS}: train={acc:.2f}%, val={val_acc:.2f}%")

        summary, preds, labels = compute_per_class_metrics(
            model, val_loader, class_names, DEVICE)

        print(f"\n  Results for {ds}:")
        print(f"    Accuracy:        {summary['accuracy']:.2f}%")
        print(f"    Macro Precision: {summary['macro_precision']:.4f}")
        print(f"    Macro Recall:    {summary['macro_recall']:.4f}")
        print(f"    Macro F1:        {summary['macro_f1']:.4f}")
        print(f"    Weighted F1:     {summary['weighted_f1']:.4f}")

        print(f"\n    {'Class':<40} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>6}")
        print(f"    {'-'*64}")
        for cls_name, m in summary["per_class"].items():
            short_name = cls_name[:38]
            print(f"    {short_name:<40} {m['precision']:.4f} {m['recall']:.4f} {m['f1']:.4f} {m['support']:>5}")

        out_path = os.path.join(RESULTS_DIR, f"metrics_{ds}.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Saved: {out_path}")

        clear_gpu()


# ===================================================================
#  2. Confusion Matrix
# ===================================================================
def run_confusion_matrix(datasets=None):
    """Generate and save confusion matrices."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  Install matplotlib and seaborn: pip install matplotlib seaborn")
        return

    if datasets is None:
        datasets = ["PlantVillageTomato", "DeepWeeds", "IP102"]

    print(f"\n{'='*70}")
    print(f"  Confusion Matrix Generation")
    print(f"{'='*70}")

    preprocess = get_clip_preprocess()

    for ds in datasets:
        print(f"\n  Dataset: {ds}")
        set_seed(42)
        train_loader, val_loader, class_names, num_classes = build_loaders(ds, preprocess)

        onto_sim = compute_similarity_matrix(ds, class_names, num_classes).to(DEVICE)
        model = AgroOntoVLM(num_classes, class_names, device=DEVICE).to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.get_param_groups(config.LR_LORA, config.LR_HEAD,
                                   config.LR_DELTA, config.LR_LOGIT_SCALE),
            weight_decay=config.WEIGHT_DECAY)
        scaler = GradScaler()

        lambda_onto = 0.05
        lambda_ce = 1.0 - config.LAMBDA_UNIFORM - lambda_onto

        print(f"  Training {config.EPOCHS} epochs...")
        for epoch in range(1, config.EPOCHS + 1):
            train_one_epoch(model, train_loader, optimizer, scaler, onto_sim,
                           num_classes, lambda_ce, config.LAMBDA_UNIFORM, lambda_onto,
                           config.LAMBDA_PROTO, DEVICE)

        _, preds, labels = compute_per_class_metrics(
            model, val_loader, class_names, DEVICE)

        cm = np.zeros((num_classes, num_classes), dtype=int)
        for p, l in zip(preds, labels):
            cm[l][p] += 1

        fig_size = max(8, num_classes * 0.4)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        if num_classes <= 20:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names, ax=ax)
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(fontsize=8)
        else:
            sns.heatmap(cm, cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

        ax.set_title(f'Confusion Matrix — AgroOnto-VLM on {ds}')
        plt.tight_layout()

        out_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{ds}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved: {out_path}")

        clear_gpu()


# ===================================================================
#  3. Disease Severity Analysis (Cross-Entropy per Severity Level)
# ===================================================================
SEVERITY_MAPPING = {
    "PlantVillageTomato": {
        "healthy": ["Tomato___healthy"],
        "mild": ["Tomato___Bacterial_spot", "Tomato___Leaf_Mold",
                 "Tomato___Spider_mites Two-spotted_spider_mite",
                 "Tomato___Target_Spot"],
        "moderate": ["Tomato___Early_blight", "Tomato___Septoria_leaf_spot",
                     "Tomato___Tomato_Yellow_Leaf_Curl_Virus"],
        "severe": ["Tomato___Late_blight",
                   "Tomato___Tomato_mosaic_virus",
                   "Tomato___Bacterial_spot"],
    },
    "DeepWeeds": {
        "non_invasive": ["Negative"],
        "low_threat": ["Chinee apple", "Snake weed"],
        "moderate_threat": ["Lantana", "Parkinsonia", "Siam weed"],
        "high_threat": ["Parthenium", "Prickly acacia", "Rubber vine"],
    },
}


def run_severity_analysis(datasets=None):
    """Analyze model performance by disease severity level."""
    if datasets is None:
        datasets = ["PlantVillageTomato", "DeepWeeds"]

    print(f"\n{'='*70}")
    print(f"  Disease Severity Analysis (Cross-Entropy per Level)")
    print(f"{'='*70}")

    preprocess = get_clip_preprocess()
    all_severity_results = {}

    for ds in datasets:
        if ds not in SEVERITY_MAPPING:
            print(f"\n  Skipping {ds} (no severity mapping defined)")
            continue

        print(f"\n  Dataset: {ds}")
        set_seed(42)
        train_loader, val_loader, class_names, num_classes = build_loaders(ds, preprocess)

        onto_sim = compute_similarity_matrix(ds, class_names, num_classes).to(DEVICE)
        model = AgroOntoVLM(num_classes, class_names, device=DEVICE).to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.get_param_groups(config.LR_LORA, config.LR_HEAD,
                                   config.LR_DELTA, config.LR_LOGIT_SCALE),
            weight_decay=config.WEIGHT_DECAY)
        scaler = GradScaler()

        lambda_onto = 0.05
        lambda_ce = 1.0 - config.LAMBDA_UNIFORM - lambda_onto

        print(f"  Training {config.EPOCHS} epochs...")
        for epoch in range(1, config.EPOCHS + 1):
            train_one_epoch(model, train_loader, optimizer, scaler, onto_sim,
                           num_classes, lambda_ce, config.LAMBDA_UNIFORM, lambda_onto,
                           config.LAMBDA_PROTO, DEVICE)

        severity_map = SEVERITY_MAPPING[ds]
        class_to_severity = {}
        for severity_level, cls_list in severity_map.items():
            for cls_name in cls_list:
                if cls_name in class_names:
                    class_to_severity[class_names.index(cls_name)] = severity_level

        model.eval()
        severity_losses = {level: [] for level in severity_map}
        severity_correct = {level: 0 for level in severity_map}
        severity_total = {level: 0 for level in severity_map}

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with autocast():
                    logits_cls, _, _ = model(images)

                for i in range(labels.size(0)):
                    label = labels[i].item()
                    if label not in class_to_severity:
                        continue
                    severity = class_to_severity[label]

                    ce_loss = F.cross_entropy(
                        logits_cls[i:i+1], labels[i:i+1]).item()
                    severity_losses[severity].append(ce_loss)

                    pred = logits_cls[i].argmax().item()
                    if pred == label:
                        severity_correct[severity] += 1
                    severity_total[severity] += 1

        print(f"\n  {'Severity Level':<20} {'CE Loss':>10} {'Accuracy':>10} {'Samples':>10}")
        print(f"  {'-'*55}")

        ds_results = {}
        for level in severity_map:
            if severity_total[level] > 0:
                avg_loss = np.mean(severity_losses[level])
                acc = 100.0 * severity_correct[level] / severity_total[level]
                print(f"  {level:<20} {avg_loss:>10.4f} {acc:>9.2f}% {severity_total[level]:>9}")
                ds_results[level] = {
                    "avg_ce_loss": round(float(avg_loss), 4),
                    "accuracy": round(float(acc), 2),
                    "num_samples": severity_total[level],
                }
            else:
                print(f"  {level:<20} {'N/A':>10} {'N/A':>10} {0:>9}")

        all_severity_results[ds] = ds_results
        clear_gpu()

    out_path = os.path.join(RESULTS_DIR, "severity_analysis.json")
    with open(out_path, "w") as f:
        json.dump(all_severity_results, f, indent=2)
    print(f"\n  Saved: {out_path}")


# ===================================================================
#  4. Multi-Seed Evaluation (Mean ± Std)
# ===================================================================
def run_multiseed_evaluation(datasets=None, seeds=None):
    """Run model with 3 seeds and report mean ± std."""
    if datasets is None:
        datasets = ["PlantVillageTomato", "DeepWeeds", "IP102"]
    if seeds is None:
        seeds = [42, 123, 456]

    print(f"\n{'='*70}")
    print(f"  Multi-Seed Evaluation (seeds: {seeds})")
    print(f"{'='*70}")

    preprocess = get_clip_preprocess()
    all_results = {}

    for ds in datasets:
        print(f"\n  Dataset: {ds}")
        seed_accs = []

        for seed in seeds:
            print(f"\n    Seed {seed}:")
            set_seed(seed)
            train_loader, val_loader, class_names, num_classes = build_loaders(ds, preprocess)

            onto_sim = compute_similarity_matrix(ds, class_names, num_classes).to(DEVICE)
            model = AgroOntoVLM(num_classes, class_names, device=DEVICE).to(DEVICE)

            optimizer = torch.optim.AdamW(
                model.get_param_groups(config.LR_LORA, config.LR_HEAD,
                                       config.LR_DELTA, config.LR_LOGIT_SCALE),
                weight_decay=config.WEIGHT_DECAY)
            scaler = GradScaler()

            lambda_onto = 0.05
            lambda_ce = 1.0 - config.LAMBDA_UNIFORM - lambda_onto

            best_acc = 0.0
            for epoch in range(1, config.EPOCHS + 1):
                train_one_epoch(model, train_loader, optimizer, scaler, onto_sim,
                               num_classes, lambda_ce, config.LAMBDA_UNIFORM, lambda_onto,
                               config.LAMBDA_PROTO, DEVICE)
                val_acc = evaluate(model, val_loader, DEVICE)
                best_acc = max(best_acc, val_acc)
                if epoch % 10 == 0 or epoch == config.EPOCHS:
                    print(f"      Epoch {epoch}/{config.EPOCHS}: val={val_acc:.2f}% (best={best_acc:.2f}%)")

            seed_accs.append(best_acc)
            print(f"    Seed {seed} best: {best_acc:.2f}%")
            clear_gpu()

        mean_acc = np.mean(seed_accs)
        std_acc = np.std(seed_accs)
        print(f"\n  {ds} Result: {mean_acc:.2f}% +/- {std_acc:.2f}%")
        print(f"  Individual: {[f'{a:.2f}%' for a in seed_accs]}")

        all_results[ds] = {
            "seeds": seeds,
            "accuracies": [round(a, 2) for a in seed_accs],
            "mean": round(float(mean_acc), 2),
            "std": round(float(std_acc), 2),
            "result_string": f"{mean_acc:.2f} +/- {std_acc:.2f}",
        }

    out_path = os.path.join(RESULTS_DIR, "multiseed_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {out_path}")

    print(f"\n  {'='*50}")
    print(f"  Multi-Seed Summary (for paper table):")
    print(f"  {'Dataset':<25} {'Mean ± Std':>20}")
    print(f"  {'-'*48}")
    for ds, r in all_results.items():
        print(f"  {ds:<25} {r['result_string']:>20}")
    print(f"  {'='*50}")


# ===================================================================
#  5. Ontology Similarity Heatmap
# ===================================================================
def run_ontology_heatmap(datasets=None):
    """Visualize ontology similarity matrices as heatmaps."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  Install matplotlib and seaborn: pip install matplotlib seaborn")
        return

    if datasets is None:
        datasets = ["PlantVillageTomato", "DeepWeeds"]

    print(f"\n{'='*70}")
    print(f"  Ontology Similarity Heatmap")
    print(f"{'='*70}")

    preprocess = get_clip_preprocess()

    for ds in datasets:
        print(f"\n  Dataset: {ds}")
        _, _, class_names, num_classes = build_loaders(ds, preprocess)
        sim_matrix = compute_similarity_matrix(ds, class_names, num_classes).numpy()

        fig_size = max(8, num_classes * 0.5)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        short_names = [n.replace("Tomato___", "").replace("_", " ")[:25] for n in class_names]

        sns.heatmap(sim_matrix, annot=(num_classes <= 20), fmt='.2f',
                   cmap='YlOrRd', xticklabels=short_names,
                   yticklabels=short_names, ax=ax,
                   vmin=0, vmax=1, square=True)

        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
        ax.set_title(f'Ontology Similarity Matrix — {ds}')
        plt.tight_layout()

        out_path = os.path.join(RESULTS_DIR, f"ontology_heatmap_{ds}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved: {out_path}")


# ===================================================================
#  6. Training Curves
# ===================================================================
def run_training_curves():
    """Plot training loss and accuracy curves from saved results."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Install matplotlib: pip install matplotlib")
        return

    print(f"\n{'='*70}")
    print(f"  Training Curves")
    print(f"{'='*70}")

    summary_path = os.path.join(config.BASE_DIR, "results", "all_methods",
                                 "all_methods_summary.json")
    if not os.path.exists(summary_path):
        print("  No results found. Run experiments first.")
        return

    with open(summary_path) as f:
        data = json.load(f)

    excluded_methods = {"CLIP Zero-Shot", "ResNet-50 Linear Probe", "ViT-B/16 Linear Probe"}
    proposed = [e for e in data if e.get("method") == "AgroOnto-VLM (Proposed)"
                and e.get("method") not in excluded_methods]

    for entry in proposed:
        ds = entry["dataset"]
        history = entry.get("history", {})
        if not history.get("train_loss"):
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        epochs = list(range(1, len(history["train_loss"]) + 1))

        ax1.plot(epochs, history["train_loss"], 'b-', linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Training Loss")
        ax1.set_title(f"Training Loss — {ds}")
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, history.get("train_acc", []), 'b-', label="Train", linewidth=2)
        ax2.plot(epochs, history.get("val_acc", []), 'r-', label="Val", linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title(f"Accuracy — {ds}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(RESULTS_DIR, f"training_curves_{ds}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved: {out_path}")


# ===================================================================
#  7. t-SNE Feature Visualization
# ===================================================================
def run_tsne_visualization(datasets=None):
    """t-SNE visualization of learned features."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        print("  Install matplotlib and scikit-learn: pip install matplotlib scikit-learn")
        return

    if datasets is None:
        datasets = ["PlantVillageTomato", "DeepWeeds"]

    print(f"\n{'='*70}")
    print(f"  t-SNE Feature Visualization")
    print(f"{'='*70}")

    preprocess = get_clip_preprocess()

    for ds in datasets:
        print(f"\n  Dataset: {ds}")
        set_seed(42)
        train_loader, val_loader, class_names, num_classes = build_loaders(ds, preprocess)

        onto_sim = compute_similarity_matrix(ds, class_names, num_classes).to(DEVICE)
        model = AgroOntoVLM(num_classes, class_names, device=DEVICE).to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.get_param_groups(config.LR_LORA, config.LR_HEAD,
                                   config.LR_DELTA, config.LR_LOGIT_SCALE),
            weight_decay=config.WEIGHT_DECAY)
        scaler = GradScaler()

        lambda_onto = 0.05
        lambda_ce = 1.0 - config.LAMBDA_UNIFORM - lambda_onto

        print(f"  Training {config.EPOCHS} epochs...")
        for epoch in range(1, config.EPOCHS + 1):
            train_one_epoch(model, train_loader, optimizer, scaler, onto_sim,
                           num_classes, lambda_ce, config.LAMBDA_UNIFORM, lambda_onto,
                           config.LAMBDA_PROTO, DEVICE)

        model.eval()
        all_features, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                with autocast():
                    features = model.encode_image(images)
                all_features.append(features.cpu().numpy())
                all_labels.extend(labels.numpy())

        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.array(all_labels)

        max_samples = min(3000, len(all_labels))
        if len(all_labels) > max_samples:
            idx = np.random.choice(len(all_labels), max_samples, replace=False)
            all_features = all_features[idx]
            all_labels = all_labels[idx]

        print(f"  Running t-SNE on {len(all_labels)} samples...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings = tsne.fit_transform(all_features)

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1],
                           c=all_labels, cmap='tab20', s=10, alpha=0.7)

        if num_classes <= 20:
            short_names = [n.replace("Tomato___", "").replace("_", " ")[:20] for n in class_names]
            legend = ax.legend(scatter.legend_elements(num=num_classes)[0],
                             short_names, loc="best", fontsize=7)
            ax.add_artist(legend)

        ax.set_title(f't-SNE Feature Visualization — AgroOnto-VLM on {ds}')
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        plt.tight_layout()

        out_path = os.path.join(RESULTS_DIR, f"tsne_{ds}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved: {out_path}")

        clear_gpu()


# ===================================================================
#  8. Grad-CAM Visualization Grid
# ===================================================================
def run_gradcam(datasets=None):
    """Generate Grad-CAM attention heatmap grids for each dataset."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("  Install matplotlib: pip install matplotlib")
        return

    if datasets is None:
        datasets = ["PlantVillageTomato", "DeepWeeds", "IP102"]

    print(f"\n{'='*70}")
    print(f"  Grad-CAM Visualization Grid")
    print(f"{'='*70}")

    preprocess = get_clip_preprocess()

    for ds in datasets:
        print(f"\n  Dataset: {ds}")
        set_seed(42)
        train_loader, val_loader, class_names, num_classes = build_loaders(ds, preprocess)

        onto_sim = compute_similarity_matrix(ds, class_names, num_classes).to(DEVICE)
        model = AgroOntoVLM(num_classes, class_names, device=DEVICE).to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.get_param_groups(config.LR_LORA, config.LR_HEAD,
                                   config.LR_DELTA, config.LR_LOGIT_SCALE),
            weight_decay=config.WEIGHT_DECAY)
        scaler = GradScaler()

        lambda_onto = 0.05
        lambda_ce = 1.0 - config.LAMBDA_UNIFORM - lambda_onto

        print(f"  Training {config.EPOCHS} epochs...")
        for epoch in range(1, config.EPOCHS + 1):
            train_one_epoch(model, train_loader, optimizer, scaler, onto_sim,
                           num_classes, lambda_ce, config.LAMBDA_UNIFORM, lambda_onto,
                           config.LAMBDA_PROTO, DEVICE)

        model.eval()

        target_layer = model.clip.visual.transformer.resblocks[-1].ln_1
        activations = {}
        gradients = {}

        def save_activation(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook

        def save_gradient(name):
            def hook(module, input, output):
                if output.requires_grad:
                    output.register_hook(lambda grad: gradients.update({name: grad.detach()}))
            return hook

        handle_fwd = target_layer.register_forward_hook(save_activation("target"))
        handle_bwd = target_layer.register_forward_hook(save_gradient("target"))

        num_samples = min(4, len(val_loader.dataset))
        sample_indices = random.sample(range(len(val_loader.dataset)), num_samples)

        fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        denorm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        denorm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

        for idx, sample_idx in enumerate(sample_indices):
            img_tensor, label = val_loader.dataset[sample_idx]
            img_input = img_tensor.unsqueeze(0).to(DEVICE).requires_grad_(True)

            activations.clear()
            gradients.clear()

            logits_cls, _, _ = model(img_input)
            pred_class = logits_cls.argmax(dim=-1).item()
            score = logits_cls[0, pred_class]
            model.zero_grad()
            score.backward()

            if "target" in activations and "target" in gradients:
                act = activations["target"]
                grad = gradients["target"]

                if act.dim() == 3:
                    act = act[1:, 0, :]
                    grad = grad[1:, 0, :]

                weights = grad.mean(dim=0)
                cam = (act * weights).sum(dim=-1)
                cam = torch.relu(cam)

                side = int(cam.shape[0] ** 0.5)
                if side * side == cam.shape[0]:
                    cam = cam.reshape(side, side)
                else:
                    side = int(np.ceil(cam.shape[0] ** 0.5))
                    cam_padded = torch.zeros(side * side, device=cam.device)
                    cam_padded[:cam.shape[0]] = cam
                    cam = cam_padded.reshape(side, side)

                cam = cam.cpu().numpy()
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

                import cv2
                cam_resized = cv2.resize(cam, (224, 224))
            else:
                cam_resized = np.zeros((224, 224))

            img_show = img_tensor.cpu() * denorm_std + denorm_mean
            img_show = img_show.clamp(0, 1).permute(1, 2, 0).numpy()

            import cv2
            img_resized = cv2.resize(img_show, (224, 224))

            heatmap = cm.jet(cam_resized)[:, :, :3]
            overlay = 0.5 * img_resized + 0.5 * heatmap

            short_name = class_names[label].replace("Tomato___", "").replace("_", " ")[:25]
            pred_name = class_names[pred_class].replace("Tomato___", "").replace("_", " ")[:25]

            axes[idx, 0].imshow(img_resized)
            axes[idx, 0].set_title(f"Original: {short_name}", fontsize=9)
            axes[idx, 0].axis('off')

            axes[idx, 1].imshow(overlay)
            axes[idx, 1].set_title(f"Grad-CAM: {pred_name}", fontsize=9)
            axes[idx, 1].axis('off')

        handle_fwd.remove()
        handle_bwd.remove()

        plt.suptitle(f'Grad-CAM Attention Maps — AgroOnto-VLM on {ds}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        out_path = os.path.join(RESULTS_DIR, f"gradcam_grid_{ds}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved: {out_path}")

        clear_gpu()


# ===================================================================
#  9. Published Methods Comparison Bar Chart
# ===================================================================
def run_comparison_barchart():
    """Bar chart: AgroOnto-VLM vs all published methods."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Install matplotlib: pip install matplotlib")
        return

    print(f"\n{'='*70}")
    print(f"  Published Methods Comparison Bar Chart")
    print(f"{'='*70}")

    pub_dir = os.path.join(config.BASE_DIR, "results", "published_methods")
    summary_path = os.path.join(pub_dir, "comparison_summary.json")
    if not os.path.exists(summary_path):
        print("  No published results found. Run published methods first.")
        return

    with open(summary_path) as f:
        data = json.load(f)

    our = data.get("our_results", {})
    published = data.get("published_repro", {})

    methods = []
    ours_acc = []
    repro_acc = []

    labels_map = {
        "P1": ("DINOv2+LoRA\n(DeepWeeds)", "DeepWeeds"),
        "P2": ("Swin-AARNet\n(IP102)", "IP102"),
        "P3": ("FFM+MAM\n(IP102)", "IP102"),
        "P4": ("CT-PestNet\n(IP102)", "IP102"),
        "P5": ("CNN-Stacking\n(PV-Tomato)", "PlantVillageTomato"),
    }

    for pid, (label, ds) in labels_map.items():
        if pid in published:
            methods.append(label)
            our_val = our.get(ds, our.get("PlantVillage", 0))
            ours_acc.append(our_val)
            repro_acc.append(published[pid].get("acc", 0))

    if not methods:
        print("  No completed methods to plot.")
        return

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, ours_acc, width, label='AgroOnto-VLM (Ours)',
                   color='#2196F3', edgecolor='white')
    bars2 = ax.bar(x + width/2, repro_acc, width, label='Published Method (Repro)',
                   color='#FF9800', edgecolor='white')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('AgroOnto-VLM vs Published SOTA Methods', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.legend(fontsize=11)
    ax.set_ylim(bottom=max(0, min(ours_acc + repro_acc) - 10))
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1:
        ax.annotate(f'{bar.get_height():.1f}%',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.annotate(f'{bar.get_height():.1f}%',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "comparison_barchart.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# ===================================================================
#  9. Ablation Study Bar Chart (M1-M4)
# ===================================================================
def run_ablation_barchart():
    """Bar chart for M1-M4 ablation results per dataset."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Install matplotlib: pip install matplotlib")
        return

    print(f"\n{'='*70}")
    print(f"  Ablation Study Bar Chart (M1-M4)")
    print(f"{'='*70}")

    summary_path = os.path.join(config.BASE_DIR, "results", "all_methods",
                                 "all_methods_summary.json")
    if not os.path.exists(summary_path):
        print("  No results found. Run all_methods first.")
        return

    with open(summary_path) as f:
        data = json.load(f)

    excluded_methods = {"CLIP Zero-Shot", "ResNet-50 Linear Probe", "ViT-B/16 Linear Probe"}

    for dataset in ["PlantVillage", "DeepWeeds", "IP102"]:
        entries = [e for e in data if e.get("dataset") == dataset
                   and e.get("method") not in excluded_methods]

        if not entries:
            continue

        methods = [e["method"].replace(" (Proposed)", "\n(Proposed)")
                   .replace(" Ablation", "\nAblation")
                   .replace(" Linear Probe", "\nLinear Probe")
                   .replace(" Baseline", "\nBaseline")
                   for e in entries]
        accs = [e.get("best_val_acc", 0) for e in entries]

        colors = ['#4CAF50' if '(Proposed)' in m else '#2196F3' if 'Baseline' in m
                  else '#FF9800' if 'Ablation' in m else '#9E9E9E'
                  for m in methods]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(methods)), accs, color=colors, edgecolor='white')

        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'Ablation Study — {dataset}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=8, ha='center')
        ax.set_ylim(bottom=max(0, min(accs) - 10))
        ax.grid(axis='y', alpha=0.3)

        for bar in bars:
            ax.annotate(f'{bar.get_height():.1f}%',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        out_path = os.path.join(RESULTS_DIR, f"ablation_barchart_{dataset}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved: {out_path}")


# ===================================================================
#  10. Parameter Efficiency Plot (Accuracy vs Params)
# ===================================================================
def run_param_efficiency():
    """Scatter plot: accuracy vs trainable parameters for all methods."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Install matplotlib: pip install matplotlib")
        return

    print(f"\n{'='*70}")
    print(f"  Parameter Efficiency Plot")
    print(f"{'='*70}")

    summary_path = os.path.join(config.BASE_DIR, "results", "all_methods",
                                 "all_methods_summary.json")
    pub_dir = os.path.join(config.BASE_DIR, "results", "published_methods")

    points = []

    excluded_methods = {"CLIP Zero-Shot", "ResNet-50 Linear Probe", "ViT-B/16 Linear Probe"}

    if os.path.exists(summary_path):
        with open(summary_path) as f:
            data = json.load(f)
        for e in data:
            if e.get("trainable_params", 0) > 0 and e.get("method") not in excluded_methods:
                points.append({
                    "method": e["method"].replace(" (Proposed)", "*"),
                    "dataset": e["dataset"],
                    "acc": e.get("best_val_acc", 0),
                    "params": e["trainable_params"],
                    "group": "ours" if "Proposed" in e.get("method", "") else "ablation",
                })

    for pid in ["P1", "P2", "P3", "P4", "P5"]:
        ds_map = {"P1": "DeepWeeds", "P2": "IP102", "P3": "IP102",
                  "P4": "IP102", "P5": "PlantVillageTomato"}
        name_map = {"P1": "DINOv2+LoRA", "P2": "Swin-AARNet", "P3": "FFM+MAM",
                    "P4": "CT-PestNet", "P5": "CNN-Stacking"}
        fpath = os.path.join(pub_dir, f"{pid}_{ds_map[pid]}.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                pdata = json.load(f)
            points.append({
                "method": name_map[pid],
                "dataset": ds_map[pid],
                "acc": pdata.get("best_val_acc", 0),
                "params": pdata.get("trainable_params", 0),
                "group": "published",
            })

    if not points:
        print("  No data to plot.")
        return

    for dataset in ["PlantVillage", "DeepWeeds", "IP102"]:
        ds_points = [p for p in points if p["dataset"] == dataset]
        if len(ds_points) < 2:
            continue

        fig, ax = plt.subplots(figsize=(10, 7))

        for p in ds_points:
            color = '#4CAF50' if p["group"] == "ours" else \
                    '#F44336' if p["group"] == "published" else '#2196F3'
            marker = '*' if p["group"] == "ours" else \
                     'D' if p["group"] == "published" else 'o'
            size = 200 if p["group"] == "ours" else 120

            ax.scatter(p["params"], p["acc"], c=color, marker=marker,
                      s=size, zorder=5, edgecolors='white', linewidth=1)
            ax.annotate(p["method"], (p["params"], p["acc"]),
                       xytext=(8, 5), textcoords="offset points",
                       fontsize=8, ha='left')

        ax.set_xlabel('Trainable Parameters', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'Parameter Efficiency — {dataset}', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='*', color='w', markerfacecolor='#4CAF50',
                   markersize=15, label='AgroOnto-VLM (Ours)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3',
                   markersize=10, label='Ablation'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='#F44336',
                   markersize=10, label='Published'),
        ]
        ax.legend(handles=legend_elements, fontsize=10)

        plt.tight_layout()
        out_path = os.path.join(RESULTS_DIR, f"param_efficiency_{dataset}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved: {out_path}")


# ===================================================================
#  Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="AgroOnto-VLM: Post-Training Analysis & Severity Study")
    parser.add_argument("--analysis", type=str, default=None,
                        choices=["metrics", "confusion", "severity",
                                 "multiseed", "heatmap", "curves", "tsne",
                                 "gradcam", "barchart", "ablation", "efficiency"],
                        help="Run only this analysis")
    args = parser.parse_args()

    print("=" * 70)
    print("  AgroOnto-VLM — Post-Training Analysis")
    print(f"  Device: {DEVICE}")
    if DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name}  |  VRAM: {props.total_memory / 1024**3:.1f} GB")
    print(f"  Results: {RESULTS_DIR}")
    print("=" * 70)

    ANALYSIS_MAP = {
        "metrics":    ("Per-Class Metrics",              run_metrics_analysis),
        "confusion":  ("Confusion Matrices",             run_confusion_matrix),
        "severity":   ("Severity Analysis",              run_severity_analysis),
        "multiseed":  ("Multi-Seed Evaluation",          run_multiseed_evaluation),
        "heatmap":    ("Ontology Heatmap",               run_ontology_heatmap),
        "curves":     ("Training Curves",                run_training_curves),
        "tsne":       ("t-SNE Visualization",            run_tsne_visualization),
        "gradcam":    ("Grad-CAM Attention Grid",        run_gradcam),
        "barchart":   ("Published Comparison Bar Chart", run_comparison_barchart),
        "ablation":   ("Ablation Study Bar Chart",       run_ablation_barchart),
        "efficiency": ("Parameter Efficiency Plot",      run_param_efficiency),
    }

    if args.analysis:
        name, fn = ANALYSIS_MAP[args.analysis]
        print(f"\n  Running: {name}")
        fn()
    else:
        for key, (name, fn) in ANALYSIS_MAP.items():
            print(f"\n  Running: {name}")
            try:
                fn()
            except Exception as e:
                print(f"  ERROR in {name}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"  All analyses complete! Results in: {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
