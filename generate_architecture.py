"""
Generate AgroOnto-VLM Architecture Diagram for Q1 journal paper.
No GPU or model needed — pure matplotlib drawing.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

fig, ax = plt.subplots(1, 1, figsize=(20, 10))
ax.set_xlim(0, 20)
ax.set_ylim(0, 10)
ax.axis('off')

# ── Colors ───────────────────────────────────────────────────────
C_INPUT  = '#E8F4FD'
C_CLIP   = '#D4E6F1'
C_LORA   = '#FDEBD0'
C_FEAT   = '#D5F5E3'
C_CLASS  = '#D4E6F1'
C_ONTO   = '#FADBD8'
C_LOSS   = '#FEF9E7'
C_OUT    = '#D5F5E3'
C_BDR    = '#2C3E50'

def box(x, y, w, h, color, label='', fsz=10, bold=True, sub=None, sub_sz=8):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                       facecolor=color, edgecolor=C_BDR, linewidth=1.5, zorder=3)
    ax.add_patch(b)
    if label:
        fw = 'bold' if bold else 'normal'
        ty = y + h/2 + (0.18 if sub else 0)
        ax.text(x+w/2, ty, label, ha='center', va='center',
                fontsize=fsz, fontweight=fw, zorder=5)
    if sub:
        ax.text(x+w/2, y+h/2-0.28, sub, ha='center', va='center',
                fontsize=sub_sz, fontstyle='italic', color='#555', zorder=5)

def arrow(x1, y1, x2, y2, color=C_BDR, lw=2.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw), zorder=2)

# ═══════════════════════════════════════════════════════════════
#  TITLE
# ═══════════════════════════════════════════════════════════════
ax.text(10, 9.5, 'AgroOnto-VLM: System Architecture',
        ha='center', va='center', fontsize=18, fontweight='bold')

# ═══════════════════════════════════════════════════════════════
#  TOP ROW: Main Pipeline  (y ~ 6.0 - 8.5)
# ═══════════════════════════════════════════════════════════════

# 1. Input Image
box(0.3, 6.2, 2.0, 2.0, C_INPUT, 'Input\nImage', fsz=12)
ax.text(1.3, 6.0, 'PlantVillage | DeepWeeds | IP102',
        ha='center', va='top', fontsize=7.5, color='#666')

# Arrow → CLIP
arrow(2.3, 7.2, 3.2, 7.2)

# 2. CLIP ViT-B/32 block (big)
box(3.2, 5.5, 3.2, 3.2, C_CLIP)
ax.text(4.8, 8.4, 'CLIP ViT-B/32', ha='center', va='center',
        fontsize=13, fontweight='bold', zorder=5)
ax.text(4.8, 8.0, '(Image Encoder — Frozen)', ha='center', va='center',
        fontsize=8.5, fontstyle='italic', color='#555', zorder=5)

# Transformer layers inside
layers = ['Patch Embedding', 'Multi-Head Self-Attention', 'Feed Forward Network']
for i, lbl in enumerate(layers):
    ly = 7.2 - i * 0.7
    inner = FancyBboxPatch((3.5, ly), 2.6, 0.55, boxstyle="round,pad=0.05",
                           facecolor='white', edgecolor='#5DADE2', linewidth=1.0, zorder=4)
    ax.add_patch(inner)
    ax.text(4.8, ly+0.275, lbl, ha='center', va='center', fontsize=8, zorder=5)
    if i < 2:
        arrow(4.8, ly, 4.8, ly+0.55, color='#5DADE2', lw=1.0)

# 3. LoRA Adapter (to the right of CLIP)
box(6.8, 6.5, 1.5, 1.8, C_LORA, 'LoRA\nAdapter', fsz=10)
ax.text(7.55, 6.35, 'rank r = 8', ha='center', va='top',
        fontsize=7.5, color='#D35400', fontstyle='italic')

# Dashed border around LoRA
lora_dash = FancyBboxPatch((6.65, 6.3), 1.8, 2.3, boxstyle="round,pad=0.08",
                           facecolor='none', edgecolor='#E67E22',
                           linewidth=1.8, linestyle='--', zorder=3)
ax.add_patch(lora_dash)
ax.text(7.55, 8.7, 'Trainable', ha='center', va='bottom',
        fontsize=9, color='#E67E22', fontweight='bold')
ax.text(7.55, 8.35, 'ΔW = A · B', ha='center', va='bottom',
        fontsize=8, color='#E67E22', fontstyle='italic')

# Connection: CLIP → LoRA (side arrow)
arrow(6.4, 7.2, 6.8, 7.2, color='#E67E22', lw=1.5)

# Arrow: LoRA → Features
arrow(8.3, 7.2, 9.2, 7.2)

# 4. Image Features
box(9.2, 6.3, 2.0, 1.8, C_FEAT, 'Image\nFeatures', fsz=11, sub='512-d vector')

# Arrow → Classifier
arrow(11.2, 7.2, 12.2, 7.2)

# 5. Linear Classifier
box(12.2, 6.3, 2.2, 1.8, C_CLASS, 'Linear\nClassifier', fsz=11, sub='K classes')

# Arrow → Output
arrow(14.4, 7.2, 15.4, 7.2)

# 6. Predicted Class
box(15.4, 6.3, 2.2, 1.8, C_OUT, 'Predicted\nClass', fsz=11, sub='ŷ = argmax(p)')

# ═══════════════════════════════════════════════════════════════
#  BOTTOM ROW: Ontology + Loss  (y ~ 1.0 - 5.0)
# ═══════════════════════════════════════════════════════════════

# Arrow: Classifier → Loss (down)
arrow(13.3, 6.3, 13.3, 5.0)
ax.text(13.55, 5.65, 'p(y|x)', ha='left', va='center',
        fontsize=9, color=C_BDR, fontstyle='italic')

# 7. Ontology Similarity Matrix
box(1.0, 1.5, 4.5, 3.3, C_ONTO)
ax.text(3.25, 4.5, 'Ontology Similarity Matrix', ha='center', va='center',
        fontsize=11, fontweight='bold', zorder=5)
ax.text(3.25, 4.1, 'S (K × K)', ha='center', va='center',
        fontsize=10, fontstyle='italic', color='#555', zorder=5)

# Mini heatmap
hm = np.array([[1.0,0.7,0.3,0.1,0.0],
               [0.7,1.0,0.5,0.2,0.1],
               [0.3,0.5,1.0,0.6,0.2],
               [0.1,0.2,0.6,1.0,0.4],
               [0.0,0.1,0.2,0.4,1.0]])
s = 0.42
hm_x0, hm_y0 = 1.8, 2.2
for i in range(5):
    for j in range(5):
        c = plt.cm.YlOrRd(hm[i,j]*0.8 + 0.1)
        r = plt.Rectangle((hm_x0+j*s, hm_y0+(4-i)*s), s, s,
                           facecolor=c, edgecolor='white', linewidth=0.8, zorder=4)
        ax.add_patch(r)

# Facet labels
ax.text(3.25, 1.95, 'Seven Facets:', ha='center', va='center',
        fontsize=8, fontweight='bold', color='#333', zorder=5)
ax.text(3.25, 1.65, 'crop · organ · pathogen_type · order · family · genus · symptom',
        ha='center', va='center', fontsize=7.5, color='#666', zorder=5)

# Arrow: Ontology → Loss
arrow(5.5, 3.2, 8.5, 3.5)

# 8. Hybrid Loss Function
box(8.5, 2.0, 5.5, 2.8, C_LOSS)
ax.text(11.25, 4.5, 'Hybrid Loss Function', ha='center', va='center',
        fontsize=12, fontweight='bold', zorder=5)
ax.text(11.25, 3.6,
        r'$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \cdot \mathcal{L}_{onto}$',
        ha='center', va='center', fontsize=15, zorder=5)
ax.text(9.8, 2.7, 'Cross-Entropy\nLoss', ha='center', va='center',
        fontsize=9, color='#1A5276', zorder=5)
ax.text(12.7, 2.7, 'Ontology-Guided\nLoss', ha='center', va='center',
        fontsize=9, color='#C0392B', zorder=5)

# Arrow: Loss → Backprop (up to classifier)
arrow(11.25, 4.8, 11.25, 6.3, color='#E74C3C', lw=2.2)
ax.text(11.5, 5.55, 'Backprop', ha='left', va='center',
        fontsize=9, color='#E74C3C', fontweight='bold')

# ═══════════════════════════════════════════════════════════════
#  Legend
# ═══════════════════════════════════════════════════════════════
legend_data = [
    (C_CLIP, 'Frozen Backbone'),
    (C_LORA, 'Trainable (LoRA)'),
    (C_ONTO, 'Ontology Knowledge'),
    (C_LOSS, 'Loss Computation'),
    (C_OUT,  'Output'),
]
for i, (c, lbl) in enumerate(legend_data):
    lx, ly = 16.5, 5.0 - i*0.55
    r = plt.Rectangle((lx, ly), 0.35, 0.35, facecolor=c,
                       edgecolor=C_BDR, linewidth=1.0, zorder=4)
    ax.add_patch(r)
    ax.text(lx+0.5, ly+0.175, lbl, ha='left', va='center', fontsize=9, zorder=4)

plt.savefig('results/analysis/architecture_diagram.png', dpi=300,
            bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("[OK] results/analysis/architecture_diagram.png")
