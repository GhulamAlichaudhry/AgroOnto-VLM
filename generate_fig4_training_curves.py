"""
Generate Figure 4: Training Convergence Curves for IEEE Q1 journal paper.
2 rows x 3 columns: top=loss, bottom=accuracy, columns=datasets.
IEEE double-column width: 7.16 inches, Times New Roman 8pt, 300 DPI.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import json

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.06,
})

OUT = os.path.join("results", "analysis")
os.makedirs(OUT, exist_ok=True)

with open("results/all_methods/all_methods_summary.json") as f:
    all_data = json.load(f)

datasets = [
    ("PlantVillage", "PlantVillageTomato", 21, 99.55),
    ("DeepWeeds",    "DeepWeeds",          15, 97.02),
    ("IP102",        "IP102",               9, 76.60),
]

proposed = {}
for entry in all_data:
    if entry["method"] == "AgroOnto-VLM (Proposed)":
        proposed[entry["dataset"]] = entry

C_TRAIN = '#1a5276'
C_VAL   = '#c0392b'

fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.8))
plt.subplots_adjust(hspace=0.45, wspace=0.38)

labels_top = ['(a)', '(b)', '(c)']
labels_bot = ['(d)', '(e)', '(f)']

for col, (ds_key, ds_name, best_ep, best_acc) in enumerate(datasets):
    d = proposed[ds_key]
    epochs = d["history"]["epoch"]
    train_loss = d["history"]["train_loss"]
    train_acc = d["history"]["train_acc"]
    val_acc = d["history"]["val_acc"]

    # ── Top row: Loss ───────────────────────────────────
    ax = axes[0, col]
    ax.plot(epochs, train_loss, '-', color=C_TRAIN, linewidth=1.2, label='Train')
    ax.set_xlabel('Epoch', fontsize=7, labelpad=2)
    ax.set_ylabel('Loss', fontsize=7, labelpad=2)
    ax.set_title(f'{labels_top[col]} {ds_name} — Loss',
                 fontsize=8, fontweight='bold', pad=4)
    ax.set_xlim(1, 30)
    ax.set_ylim(0, max(train_loss) * 1.1)
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.legend(loc='upper right', framealpha=0.8, edgecolor='none')

    # ── Bottom row: Accuracy ────────────────────────────
    ax = axes[1, col]
    ax.plot(epochs, train_acc, '-', color=C_TRAIN, linewidth=1.2, label='Train')
    ax.plot(epochs, val_acc, '--', color=C_VAL, linewidth=1.2, label='Validation')

    best_idx = best_ep - 1
    ax.plot(best_ep, val_acc[best_idx], '*', color=C_VAL,
            markersize=8, zorder=5, markeredgecolor='black', markeredgewidth=0.4)

    y_off = 1.5 if best_acc > 90 else 2.5
    ax.annotate(f'{best_acc:.2f}%',
                xy=(best_ep, val_acc[best_idx]),
                xytext=(best_ep + 2, val_acc[best_idx] - y_off),
                fontsize=6, fontweight='bold', color=C_VAL,
                arrowprops=dict(arrowstyle='->', color=C_VAL, lw=0.6),
                zorder=6)

    ax.set_xlabel('Epoch', fontsize=7, labelpad=2)
    ax.set_ylabel('Accuracy (%)', fontsize=7, labelpad=2)
    ax.set_title(f'{labels_bot[col]} {ds_name} — Accuracy',
                 fontsize=8, fontweight='bold', pad=4)
    ax.set_xlim(1, 30)

    all_vals = train_acc + val_acc
    y_min = min(all_vals) - 3
    y_max = max(all_vals) + 2
    if y_min > 80:
        y_min = max(80, y_min)
    ax.set_ylim(y_min, min(y_max, 101))

    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.legend(loc='lower right', framealpha=0.8, edgecolor='none')

plt.savefig(os.path.join(OUT, 'fig4_training_curves.png'),
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("[OK] results/analysis/fig4_training_curves.png")
