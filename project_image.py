"""Project image for the submission form (1600x900 PNG): agents -> AI auditors -> human."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

NAVY, LIGHT, SOFT, ORANGE, BLUE = "#14213D", "#F7F5F0", "#C9D3E3", "#F2A65A", "#6FA3D2"
fig = plt.figure(figsize=(16, 9), dpi=100)
fig.patch.set_facecolor(NAVY)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")

# title block
ax.text(0.9, 7.55, "GROUP STRATEGY-PROOF MECHANISM", color=ORANGE, fontsize=20,
        fontweight="bold", family="DejaVu Sans")
ax.text(0.9, 6.35, "Can a swarm of AI agents\nbe audited honestly?", color=LIGHT,
        fontsize=40, fontweight="bold", family="DejaVu Serif", va="center", linespacing=1.15)
ax.text(0.9, 4.75, "Audit theory + evidence from a real swarm\nof 3,102 agents sharing answers",
        color=SOFT, fontsize=21, family="DejaVu Sans", va="center", linespacing=1.4)
ax.text(0.9, 0.75, "Yiping Miao · UC Berkeley · Apart Swarm Alignment Hackathon 2026",
        color=SOFT, fontsize=15, family="DejaVu Sans")

# diagram: workers (bottom) -> AI auditors (middle) -> human (top)
rng = np.random.default_rng(3)
wx = np.linspace(9.0, 14.4, 9); wy = np.full(9, 1.9) + rng.normal(0, 0.12, 9)
ax_ = np.array([9.68, 11.7, 13.72]); ay = np.full(3, 4.6)
hx, hy = 11.7, 7.2
# coalition halo around the first three workers + their auditor
ax.add_patch(FancyBboxPatch((8.6, 1.35), 2.15, 3.9, boxstyle="round,pad=0.1,rounding_size=0.4",
                            fc=ORANGE, alpha=0.12, ec=ORANGE, lw=2, ls="--"))
ax.text(8.75, 5.42, "coalition", color=ORANGE, fontsize=15, style="italic")
for i, (x, y) in enumerate(zip(wx, wy)):
    a = ax_[min(2, int(i / 3))]
    ax.add_patch(FancyArrowPatch((x, y + 0.3), (a, ay[0] - 0.42), arrowstyle="-|>",
                                 mutation_scale=12, color=SOFT, alpha=0.45, lw=1.2))
for a in ax_:
    ax.add_patch(FancyArrowPatch((a, ay[0] + 0.42), (hx, hy - 0.5), arrowstyle="-|>",
                                 mutation_scale=14, color=SOFT, alpha=0.7, lw=1.6))
# skip-level human check (dashed) to a raw report
ax.add_patch(FancyArrowPatch((hx + 0.55, hy - 0.35), (wx[-1], wy[-1] + 0.35),
                             connectionstyle="arc3,rad=-0.35", arrowstyle="-|>",
                             mutation_scale=16, color=ORANGE, lw=2.2, ls="--"))
for i, (x, y) in enumerate(zip(wx, wy)):
    ax.add_patch(Circle((x, y), 0.28, fc=ORANGE if i < 3 else BLUE, ec=LIGHT, lw=1.5))
for a in ax_:
    ax.add_patch(FancyBboxPatch((a - 0.45, ay[0] - 0.38), 0.9, 0.76,
                                boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=ORANGE if a == ax_[0] else BLUE, ec=LIGHT, lw=1.5))
ax.add_patch(Circle((hx, hy), 0.5, fc=LIGHT, ec=ORANGE, lw=3))
ax.text(hx, hy, "H", color=NAVY, fontsize=22, fontweight="bold", ha="center", va="center")
ax.text(14.8, 1.9, "agents", color=SOFT, fontsize=15, va="center")
ax.text(14.65, 4.6, "AI\nauditors", color=SOFT, fontsize=15, va="center")
ax.text(12.35, 7.65, "human", color=SOFT, fontsize=15)
ax.text(14.6, 6.25, "skip-level\ncheck", color=ORANGE, fontsize=14, ha="center")
plt.savefig("project_image.png", dpi=100, facecolor=NAVY)
print("Saved project_image.png")
