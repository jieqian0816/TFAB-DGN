"""Generate the TFAB-DGN architecture diagram (docs/architecture.png).

Dual-branch layout mirroring the paper:
  external graph branch (smooth regions)  ||  internal graph branch (TFAB block)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C_SPEC = "#7c5cbf"
C_EXT  = "#3b82f6"
C_MAG  = "#f59e0b"
C_TFAB = "#ef4444"
C_PH   = "#10b981"
C_FUSE = "#6366f1"
C_DARK = "#0f172a"
TXT    = "#1e293b"
ARROW  = "#64748b"

fig, ax = plt.subplots(figsize=(15, 9.5))
ax.set_xlim(0, 15.4)
ax.set_ylim(0, 9.5)
ax.axis("off")


def box(x, y, w, h, text, fc, ec=None, fs=10, tc=TXT, bold=False):
    ec = ec or fc
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3,
            fontweight="bold" if bold else "normal")


def arrow(x1, y1, x2, y2, color=ARROW, lw=1.6, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=14, linewidth=lw, color=color,
                 linestyle=ls, zorder=1,
                 connectionstyle=f"arc3,rad={rad}"))


def line(x1, y1, x2, y2, color=ARROW, lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-",
                 linewidth=lw, color=color, zorder=1))


def label(x, y, text, color, fs=11):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color=color, fontweight="bold", style="italic")


# ---------------- title ----------------
ax.text(7.7, 9.15, "TFAB-DGN: Topology-Feature-Aware Boundary Dual Graph Network",
        ha="center", va="center", fontsize=16, fontweight="bold", color=TXT)

# ---------------- input & backbone ----------------
box(0.3, 4.3, 1.6, 1.3, "HSI Input\n[H, W, B]", "#e2e8f0", ec="#94a3b8", bold=True)
box(2.3, 4.3, 2.0, 1.3, "Spectral Dim.\nReduction\n(1x1 Conv, Eq.4-5)", C_SPEC, tc="white", bold=True)
box(2.3, 2.2, 2.0, 1.2, "SLIC Superpixels\nNode Features\n(Eq.6-7)", "#cbd5e1", ec="#94a3b8", fs=9)
arrow(1.9, 4.95, 2.3, 4.95)
arrow(3.3, 4.3, 3.3, 3.4)

# ---------------- external branch (top) ----------------
label(7.0, 8.0, "Global external graph branch\n(smooth regions)", C_EXT, fs=10.5)
box(5.9, 6.5, 2.2, 1.0, "External GCN\nx3 layers (Eq.17)", C_EXT, tc="white", bold=True)
arrow(4.3, 4.95, 5.9, 6.8, color=C_EXT)

# ---------------- internal branch (bottom) ----------------
label(7.0, 2.05, "Local internal graph branch\n(boundary regions)", C_TFAB, fs=10.5)

box(4.7, 0.4, 2.2, 1.0, "Magnitude Node\nScoring (Eq.12-14)", C_MAG, tc="white", bold=True)
box(7.5, 0.4, 2.3, 1.0, "TFAB Block\nInternal GCN (Eq.18-21)", C_TFAB, tc="white", bold=True)
box(10.4, 0.4, 2.2, 1.0, "PH Reweighting\n(Eq.20)", C_PH, tc="white", bold=True)

arrow(4.3, 2.2, 4.7, 1.1, color=C_MAG)             # node features -> scoring
arrow(6.9, 0.9, 7.5, 0.9, color=C_TFAB)            # scoring -> TFAB
arrow(9.8, 0.9, 10.4, 0.9, color=C_PH)             # TFAB -> PH
arrow(11.2, 1.4, 8.9, 1.4, color=C_PH, rad=-0.35)  # PH -> TFAB (reweighted adj, arc above)

# magnitude weight map (offline prior)
box(2.0, 0.4, 2.2, 1.0, "Magnitude Weight\nMap (offline, Alg.2)", "#fde68a", ec=C_MAG, fs=9)
arrow(3.1, 1.4, 4.7, 1.1, color=C_MAG)

# ---------------- output chain (right column, vertical) ----------------
box(11.4, 6.5, 3.4, 1.1, "Feature Fusion\nmulti-scale weighted (Eq.22-24)", C_FUSE, tc="white", bold=True)
box(11.4, 4.3, 3.4, 1.0, "Classifier (Eq.25)", C_DARK, ec=C_DARK, tc="white", bold=True)
box(11.4, 2.2, 3.4, 1.0, "Prediction Map [H, W]", "#e2e8f0", ec="#94a3b8", bold=True)

arrow(8.1, 7.0, 11.4, 7.0, color=C_EXT)            # external -> fusion
arrow(13.1, 6.5, 13.1, 5.3)                        # fusion -> classifier
arrow(13.1, 4.3, 13.1, 3.2)                        # classifier -> prediction map

# internal branch output -> fusion (elbow along the right edge)
line(12.6, 0.9, 14.9, 0.9, color=C_TFAB)
line(14.9, 0.9, 14.9, 7.05, color=C_TFAB)
arrow(14.9, 7.05, 14.8, 7.05, color=C_TFAB)

# ---------------- legend note ----------------
ax.text(0.3, 0.08, "Solid arrows = feature flow;  colors denote module families "
        "(purple: spectral, blue: external GCN, amber: magnitude, red: TFAB, green: PH).",
        fontsize=8.5, color="#64748b")

plt.tight_layout()
plt.savefig("docs/architecture.png", dpi=200, bbox_inches="tight", facecolor="white")
print("saved docs/architecture.png")
