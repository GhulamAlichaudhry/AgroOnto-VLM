"""
Generate IP102 per-class F1 figure — matching PlantVillage/DeepWeeds style.
Output: results/ip102_perclass_f1.png
"""
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Times New Roman", "DejaVu Serif"],
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 8,
})

JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "results", "metrics_IP102_real.json")
OUT_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "results", "ip102_perclass_f1.png")

def get_order(cid):
    if   0  <= cid <= 17:  return "Hemiptera"
    elif 18 <= cid <= 44:  return "Lepidoptera"
    elif 45 <= cid <= 69:  return "Coleoptera"
    elif 70 <= cid <= 80:  return "Diptera"
    elif 81 <= cid <= 88:  return "Thysanoptera"
    elif 89 <= cid <= 95:  return "Acari"
    elif 96 <= cid <= 101: return "Orthoptera"

SEV_COLOR = {
    "Hemiptera":"#d62728","Lepidoptera":"#d62728","Coleoptera":"#d62728",
    "Thysanoptera":"#ff7f0e","Acari":"#ff7f0e","Orthoptera":"#ff7f0e",
    "Diptera":"#2ca02c",
}
ORDER_ABBR = {
    "Hemiptera":"Hem","Lepidoptera":"Lep","Coleoptera":"Col",
    "Diptera":"Dip","Thysanoptera":"Thy","Acari":"Aca","Orthoptera":"Ort",
}

with open(JSON_PATH) as f:
    data = json.load(f)
per_class = data["per_class"]

order_cnt = {o:0 for o in ORDER_ABBR}
rows = []
for cid in range(102):
    m     = per_class[f"pest_{cid}"]
    order = get_order(cid)
    order_cnt[order] += 1
    rows.append((f"{ORDER_ABBR[order]}-{order_cnt[order]:02d}", m["f1"], SEV_COLOR[order]))

rows_sorted = sorted(rows, key=lambda x: x[1], reverse=True)
labels = [r[0] for r in rows_sorted]
f1s    = [r[1] for r in rows_sorted]
colors = [r[2] for r in rows_sorted]
macro  = np.mean(f1s)

fig, ax = plt.subplots(figsize=(8, 22))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

y = np.arange(len(labels))
ax.barh(y, f1s, color=colors, height=0.68, edgecolor="none", zorder=3)
ax.invert_yaxis()
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=7.8)

ax.axvline(macro, color="#555555", linewidth=1.2, linestyle="--", zorder=4)
ax.text(macro + 0.005, -0.8, f"Macro-F1 = {macro:.3f}",
        fontsize=8, color="#555555", va="top", ha="left")

for i, v in enumerate(f1s):
    ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=6.5, color="#333333")

ax.set_title("IP102 Insect Pest", fontsize=13, fontweight="bold", pad=10, loc="center")
ax.text(0.01, 1.01, "(c)", transform=ax.transAxes,
        fontsize=13, fontweight="bold", va="bottom", ha="left")
ax.set_xlabel("F1 Score", fontsize=11)
ax.xaxis.grid(True, color="#e5e5e5", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#cccccc")
ax.spines["bottom"].set_color("#cccccc")
ax.tick_params(axis="both", which="both", length=0)
ax.set_xlim(0, 1.10)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {OUT_PATH}")
