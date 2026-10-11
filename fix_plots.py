"""
Fix confusion matrices and Grad-CAM grid layouts.
Regenerates improved confusion matrices from saved data.
No GPU or training needed — runs in seconds.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

try:
    import seaborn as sns
except ImportError:
    print("Installing seaborn...")
    os.system("pip install seaborn")
    import seaborn as sns

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 11,
    'axes.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

OUT = os.path.join("results", "analysis")
os.makedirs(OUT, exist_ok=True)

# ═══════════════════════════════════════════════════════════
#  DeepWeeds Confusion Matrix (9 classes)
# ═══════════════════════════════════════════════════════════
dw_names = ["Chinee apple", "Lantana", "Parkinsonia", "Parthenium",
            "Prickly acacia", "Rubber vine", "Siam weed", "Snake weed", "Negative"]

dw_cm = np.array([
    [153,  5,  1,  2,  3,  0,  1, 17, 43],
    [  1,171,  0,  0,  0,  0,  1,  8, 32],
    [  0,  0,192,  1,  2,  1,  0,  0, 10],
    [  0,  0,  0,188,  1,  1,  0,  0, 14],
    [  0,  0,  6,  2,188,  0,  0,  0, 16],
    [  0,  0,  0,  1,  0,174,  4,  2, 21],
    [  0,  0,  0,  0,  0,  0,193,  2, 20],
    [  6,  2,  0,  3,  0,  1,  2,172, 17],
    [  0,  0,  2,  6,  5,  1,  3,  6,1798],
])

fig, ax = plt.subplots(figsize=(10, 9))
sns.heatmap(dw_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=dw_names, yticklabels=dw_names, ax=ax,
            linewidths=0.5, linecolor='white',
            cbar_kws={'shrink': 0.8}, annot_kws={'size': 11})
ax.set_xlabel("Predicted", fontsize=13, labelpad=10)
ax.set_ylabel("True", fontsize=13, labelpad=10)
ax.set_title("Confusion Matrix — AgroOnto-VLM on DeepWeeds", fontsize=15, pad=15)
plt.xticks(rotation=40, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
p = os.path.join(OUT, "confusion_matrix_DeepWeeds_improved.png")
plt.savefig(p, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {p}")


# ═══════════════════════════════════════════════════════════
#  PlantVillageTomato Confusion Matrix (10 classes) — CLEAN LABELS
# ═══════════════════════════════════════════════════════════
pv_names_clean = [
    "Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold",
    "Septoria Leaf Spot", "Spider Mites", "Target Spot",
    "Yellow Leaf Curl Virus", "Mosaic Virus", "Healthy"
]

pv_cm = np.array([
    [416,  0,  0,  0,  2,  0,  0,  7,  0,  0],
    [  0,194,  2,  1,  1,  0,  2,  0,  0,  0],
    [  0,  1,376,  1,  0,  0,  0,  4,  0,  0],
    [  0,  0,  0,186,  2,  3,  0,  0,  0,  0],
    [  0,  1,  0,  0,352,  0,  0,  0,  1,  0],
    [  0,  0,  1,  0,  0,330,  3,  1,  0,  0],
    [  0,  0,  0,  0,  3,  4,273,  0,  0,  1],
    [  0,  0,  0,  0,  0,  1,  0,1070, 0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0, 74,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0,318],
])

fig, ax = plt.subplots(figsize=(11, 10))
sns.heatmap(pv_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=pv_names_clean, yticklabels=pv_names_clean, ax=ax,
            linewidths=0.5, linecolor='white',
            cbar_kws={'shrink': 0.8}, annot_kws={'size': 11})
ax.set_xlabel("Predicted", fontsize=13, labelpad=10)
ax.set_ylabel("True", fontsize=13, labelpad=10)
ax.set_title("Confusion Matrix — AgroOnto-VLM on PlantVillageTomato", fontsize=15, pad=15)
plt.xticks(rotation=40, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
p = os.path.join(OUT, "confusion_matrix_PlantVillageTomato_improved.png")
plt.savefig(p, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {p}")


# ═══════════════════════════════════════════════════════════
#  IP102 Confusion Matrix — Normalized heatmap (102 classes)
#  Read from saved .npy if available, else show placeholder
# ═══════════════════════════════════════════════════════════
npy_path = os.path.join(OUT, "confusion_matrix_IP102.npy")
if os.path.exists(npy_path):
    ip_cm = np.load(npy_path)
else:
    ip_cm = None

if ip_cm is not None:
    cm_norm = ip_cm.astype(float) / (ip_cm.sum(axis=1, keepdims=True) + 1e-8) * 100
    num_classes = ip_cm.shape[0]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_norm, cmap='Blues', ax=ax, vmin=0, vmax=100,
                cbar_kws={'shrink': 0.7, 'label': 'Accuracy (%)'})
    ax.set_xlabel("Predicted Class Index", fontsize=13, labelpad=10)
    ax.set_ylabel("True Class Index", fontsize=13, labelpad=10)
    tick_step = 10
    ticks = list(range(0, num_classes, tick_step))
    ax.set_xticks([t + 0.5 for t in ticks])
    ax.set_xticklabels(ticks, fontsize=9)
    ax.set_yticks([t + 0.5 for t in ticks])
    ax.set_yticklabels(ticks, fontsize=9)
    ax.set_title("Confusion Matrix — AgroOnto-VLM on IP102 (Normalized %)",
                 fontsize=15, pad=15)
    plt.tight_layout()
    p = os.path.join(OUT, "confusion_matrix_IP102_improved.png")
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {p}")
else:
    print("  IP102 .npy not found — skipping (will be saved on next analysis run)")
    print("  To generate it, run: python run_analysis.py confusion")


# ═══════════════════════════════════════════════════════════
#  Fix Grad-CAM Grid Spacing (if originals exist)
# ═══════════════════════════════════════════════════════════
from PIL import Image

for ds in ["PlantVillageTomato", "DeepWeeds", "IP102"]:
    src = os.path.join(OUT, f"gradcam_grid_{ds}.png")
    if not os.path.exists(src):
        continue

    img = Image.open(src)
    w, h = img.size

    canvas_h = h + 80
    canvas = Image.new('RGB', (w, canvas_h), (255, 255, 255))
    canvas.paste(img, (0, 80))

    dst = os.path.join(OUT, f"gradcam_grid_{ds}_improved.png")
    canvas.save(dst, dpi=(300, 300))
    print(f"  Saved: {dst}")


# ═══════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Fixed plots saved to:", OUT)
print("=" * 60)
print("  Improvements:")
print("    - PlantVillageTomato: Removed 'Tomato___' prefix, clean labels")
print("    - DeepWeeds: Better formatting, 300 DPI, serif fonts")
print("    - IP102: Normalized heatmap (if .npy data available)")
print("    - Grad-CAM: Fixed title overlap with padding")
print("=" * 60)
