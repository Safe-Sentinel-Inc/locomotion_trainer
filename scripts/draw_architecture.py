"""
AME-2 Architecture Diagram — clean grid layout.
Saves: docs/architecture.png
Run:   python scripts/draw_architecture.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import to_rgba

os.makedirs("docs", exist_ok=True)

# ── Colours ──────────────────────────────────────────────────────────────────
BG     = "#FAFAFA"
C_MAP  = "#2980B9"   # perception / mapping  →  blue
C_PRO  = "#D35400"   # proprioception         →  orange
C_ENC  = "#27AE60"   # AME-2 encoder          →  green
C_DEC  = "#8E44AD"   # decoder / output       →  purple
C_CRIT = "#C0392B"   # critic (train only)    →  red
C_NOTE = "#7F8C8D"   # annotation grey
TEXT   = "#2C3E50"

ALPHA  = 0.15        # box fill transparency
EALPHA = 0.10        # encoder background

FIG_W, FIG_H = 20, 13.5
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
ax.set_facecolor(BG)
fig.patch.set_facecolor(BG)

# ── Helpers ───────────────────────────────────────────────────────────────────

def rbox(ax, cx, cy, w, h, fc, ec, label, sub="",
         lfs=8.5, sfs=6.8, bold=False, ls="solid", alpha=ALPHA):
    """Rounded rectangle with optional subtitle."""
    patch = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.07",
        linewidth=1.5 if ls == "solid" else 1.0,
        linestyle=ls,
        edgecolor=ec,
        facecolor=to_rgba(fc, alpha),
        zorder=3,
    )
    ax.add_patch(patch)
    dy = h * 0.13 if sub else 0
    ax.text(cx, cy + dy, label, ha="center", va="center",
            fontsize=lfs, fontweight="bold" if bold else "normal",
            color=TEXT, zorder=5)
    if sub:
        ax.text(cx, cy - h*0.21, sub, ha="center", va="center",
                fontsize=sfs, color=TEXT, alpha=0.80, zorder=5)

def bg_box(ax, x0, y0, x1, y1, color, label="", lfs=8):
    """Background section box (dashed, very transparent)."""
    patch = FancyBboxPatch(
        (x0, y0), x1-x0, y1-y0,
        boxstyle="round,pad=0.12",
        linewidth=1.3, linestyle="--",
        edgecolor=color,
        facecolor=to_rgba(color, 0.04),
        zorder=1,
    )
    ax.add_patch(patch)
    if label:
        ax.text((x0+x1)/2, y1 - 0.18, label,
                ha="center", va="top",
                fontsize=lfs, color=color, fontweight="bold", zorder=4)

def harrow(ax, x0, x1, y, color=TEXT, lw=1.5, label="", label_va="bottom"):
    """Horizontal arrow."""
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=lw, mutation_scale=9),
                zorder=6)
    if label:
        mx = (x0+x1)/2
        dy = 0.10 if label_va == "top" else -0.12
        ax.text(mx, y + dy, label, ha="center",
                va="bottom" if label_va == "top" else "top",
                fontsize=6.5, color=color, zorder=6)

def varrow(ax, x, y0, y1, color=TEXT, lw=1.5, label="", label_ha="right"):
    """Vertical arrow."""
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=lw, mutation_scale=9),
                zorder=6)
    if label:
        my = (y0+y1)/2
        dx = -0.10 if label_ha == "right" else 0.10
        ax.text(x + dx, my, label, ha=label_ha, va="center",
                fontsize=6.5, color=color, zorder=6)

def lconnect(ax, x0, y0, x1, y1, color=TEXT, lw=1.3, label=""):
    """L-shaped connector: horizontal then vertical (or v then h)."""
    ax.plot([x0, x1], [y0, y0], color=color, lw=lw, zorder=5)
    ax.annotate("", xy=(x1, y1), xytext=(x1, y0),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=lw, mutation_scale=9),
                zorder=6)
    if label:
        ax.text((x0+x1)/2, y0+0.10, label, ha="center", va="bottom",
                fontsize=6.5, color=color, zorder=6)

def note(ax, x, y, text, color=C_NOTE, fs=6.5, ha="center", style="italic"):
    ax.text(x, y, text, ha=ha, va="center",
            fontsize=fs, color=color, fontstyle=style, zorder=5)

# ═════════════════════════════════════════════════════════════════════════════
# TITLE
# ═════════════════════════════════════════════════════════════════════════════
ax.text(FIG_W/2, FIG_H - 0.35,
        "AME-2: Attention-Based Neural Map Encoding — Policy Architecture",
        ha="center", va="center", fontsize=13.5, fontweight="bold", color=TEXT)
ax.text(FIG_W/2, FIG_H - 0.75,
        "Zhang, Klemm, Yang, Hutter (ETH Zurich RSL) · arXiv:2601.08485",
        ha="center", va="center", fontsize=8, color=C_NOTE)

# ═════════════════════════════════════════════════════════════════════════════
# HORIZONTAL BAND DEFINITIONS  (from top to bottom)
#
#  Band 1  y=9.4–10.9   MAPPING PIPELINE
#  Band 2  y=6.5–9.1    AME-2 ENCODER  (2 sub-rows inside)
#  Band 3  y=4.4–6.2    PROPRIOCEPTION
#  Band 4  y=2.8–4.1    DECODER + OUTPUT
#  Band 5  y=0.8–2.5    ASYMMETRIC CRITIC (training only)
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# BAND 1 – MAPPING PIPELINE   y_centre ≈ 10.1
# ─────────────────────────────────────────────────────────────────────────────
Y1 = 10.15
BH = 0.72   # box height
BW = 2.4    # default box width

bg_box(ax, 0.3, 9.35, 17.5, 10.90, C_MAP, "Mapping Pipeline")

# boxes  (x positions chosen so arrows are straight horizontal)
rbox(ax,  1.6, Y1, 2.0, BH, C_MAP, C_MAP, "Depth Cloud",
     "31×51 cells @ 4 cm", bold=True)
rbox(ax,  4.6, Y1, 2.4, BH, C_MAP, C_MAP, "MappingNet",
     "U-Net  (9 475 params)\nβ-NLL+TV loss  Eq.9–10")
rbox(ax,  8.0, Y1, 2.6, BH, C_MAP, C_MAP, "WTA Fusion",
     "Probabilistic WTA  Eq.6–8\nGlobal map  400×400 @ 8 cm")
rbox(ax, 11.6, Y1, 2.6, BH, C_MAP, C_MAP, "Neural Map",
     "Crop  14×36 @ 8 cm\n(elev, nx, ny, var)  4 ch",
     bold=True)

# output of neural map goes right to vertical bus
rbox(ax, 15.3, Y1, 2.2, BH, C_MAP, C_MAP, "Policy Map Input",
     "Student:  neural map  4 ch\nTeacher: GT RayCaster  3 ch")

harrow(ax,  2.6,  3.4, Y1, color=C_MAP, label="raw elev")
harrow(ax,  5.8,  6.7, Y1, color=C_MAP, label="elev + log σ²")
harrow(ax,  9.3, 10.3, Y1, color=C_MAP, label="crop + normals")
harrow(ax, 12.9, 14.2, Y1, color=C_MAP, label="4ch  14×36")

# Phase 0 tag
note(ax, 17.9, Y1 + 0.20, "Phase 0", color=C_MAP, fs=7.5, style="normal")
note(ax, 17.9, Y1 - 0.15, "Pretrain\n(no Isaac Sim)", color=C_MAP, fs=6.5)


# ─────────────────────────────────────────────────────────────────────────────
# BAND 2 – AME-2 ENCODER  (tall section, 2 internal rows)
# ─────────────────────────────────────────────────────────────────────────────
bg_box(ax, 0.3, 6.45, 17.5, 9.15, C_ENC, "AME-2 Encoder  (Fig. 3)")

# Internal layout:
#  Row A (y=8.4): Local CNN │ CoordPosEmb → Fusion MLP → K,V │ Global MLP+MaxPool
#  Row B (y=7.2): [input splice from above]   → MHA  ← Query Proj ← cat(global+prop)

EY_A = 8.35   # top sub-row
EY_B = 7.25   # bottom sub-row

# ── Row A ──
rbox(ax,  2.0, EY_A, 2.1, 0.65, C_ENC, C_ENC, "Local CNN",
     "Conv2d×2 → 64D\nper-cell feats")
rbox(ax,  4.8, EY_A, 2.0, 0.65, C_ENC, C_ENC, "CoordPosEmb",
     "MLP(2→64)\nper-cell pos encoding")
rbox(ax,  7.8, EY_A, 2.2, 0.65, C_ENC, C_ENC, "Fusion MLP",
     "Linear(128→64)\nPointwise K, V  64D", bold=True)
rbox(ax, 11.2, EY_A, 2.4, 0.65, C_ENC, C_ENC, "Global MLP",
     "Linear(64→128)\n+ MaxPool → 128D")
rbox(ax, 14.5, EY_A, 2.2, 0.65, C_ENC, C_ENC, "global_feat",
     "128D  (terrain context)")

harrow(ax,  3.05,  3.80, EY_A, color=C_ENC)
harrow(ax,  5.80,  6.70, EY_A, color=C_ENC, label="local+pe")
harrow(ax,  8.90,  9.95, EY_A, color=C_ENC, label="pointwise")
harrow(ax, 12.40, 13.40, EY_A, color=C_ENC)

# ── Row B ──
rbox(ax,  4.0, EY_B, 2.2, 0.65, C_ENC, C_ENC, "Query Proj",
     "cat(global 128, prop 128)\nMLP → Q  64D", bold=True)
rbox(ax,  7.8, EY_B, 2.2, 0.65, C_ENC, C_ENC, "MHA  (h=16)",
     "Q  64D\nK,V  64D per cell", bold=True)
rbox(ax, 11.5, EY_B, 2.4, 0.65, C_ENC, C_ENC, "cat[ wL ‖ global ]",
     "weighted_local 64D\n+ global 128D", bold=True)
rbox(ax, 14.8, EY_B, 2.0, 0.65, C_ENC, C_ENC, "map_emb",
     "192D\n= 64 + 128", bold=True)

harrow(ax,  5.10,  6.70, EY_B, color=C_ENC, label="Q  64D")
harrow(ax,  8.90,  10.3, EY_B, color=C_ENC, label="weighted\nlocal 64D")
harrow(ax, 12.70, 13.80, EY_B, color=C_ENC, label="map_emb\n192D")

# K,V  from Fusion MLP (row A) down to MHA (row B)
lconnect(ax, 7.8, EY_A - 0.33, 7.8, EY_B + 0.33, color=C_ENC, label="K,V")

# global_feat  (row A) → feeds Query Proj (row B) — vertical drop
lconnect(ax, 14.5, EY_A - 0.33, 4.0 + 1.1, EY_B + 0.33, color=C_ENC,
         label="global 128D")

# map input (top of band 2) → Local CNN and CoordPosEmb
varrow(ax, 1.6, 9.35, EY_A + 0.33, color=C_MAP, label="map\n4ch")
# shared input line to both Local CNN and CoordPosEmb
ax.plot([1.6, 4.8], [EY_A + 0.33, EY_A + 0.33],
        color=C_MAP, lw=1.4, zorder=5)
ax.annotate("", xy=(4.8, EY_A + 0.33), xytext=(4.8, EY_A + 0.33),
            arrowprops=dict(arrowstyle="->", color=C_MAP, lw=1.4))
ax.plot([4.8, 4.8], [EY_A + 0.33, EY_A + 0.33],
        color=C_MAP, lw=1.4, zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# BAND 3 – PROPRIOCEPTION ENCODER   y_centre ≈ 5.35
# ─────────────────────────────────────────────────────────────────────────────
Y3 = 5.35
bg_box(ax, 0.3, 4.40, 17.5, 6.25, C_PRO,
       "Proprioception Encoder")

rbox(ax,  1.8, Y3, 2.0, 0.72, C_PRO, C_PRO, "Prop History",
     "T=20 steps\n42D / step  (no v_base, no cmd)", bold=True)
rbox(ax,  4.8, Y3, 2.4, 0.72, C_PRO, C_PRO, "LSIO",
     "Short: last 4×42 → 168D\nLong:  Conv1d×2  → 16D\nOut:   184D")
rbox(ax,  7.9, Y3, 1.6, 0.72, C_PRO, C_PRO, "cmd_actor",
     "[clip(d_xy,2m)\nsin θ,  cos θ]  3D")
rbox(ax, 10.5, Y3, 2.2, 0.72, C_PRO, C_PRO, "Prop MLP",
     "cat(LSIO 184, cmd 3)\nLinear(187→256→128)\nprop_emb  128D", bold=True)
rbox(ax, 13.8, Y3, 2.0, 0.72, C_PRO, C_PRO, "prop_emb",
     "128D")

harrow(ax,  2.8,  3.60, Y3, color=C_PRO)
harrow(ax,  5.6, 6.60, Y3 - 0.08, color=C_PRO,
       label="LSIO 184D →\ncat(cmd 3D)")
harrow(ax,  8.7, 9.40, Y3, color=C_PRO)
harrow(ax, 11.6, 12.80, Y3, color=C_PRO, label="prop_emb 128D")

# prop_emb vertical line going up to Query Proj in encoder
# and down to Decoder
# We draw a vertical bus at x=13.8 from y3+0.36 up to EY_B-0.33
ax.plot([13.8, 13.8], [Y3 + 0.36, EY_B - 0.33],
        color=C_PRO, lw=1.4, ls="--", zorder=5)
ax.annotate("", xy=(5.10, EY_B - 0.33), xytext=(5.10, EY_B - 0.33))
# horizontal stub: x=13.8 → x=5.10 at y=EY_B-0.10
ax.plot([5.10, 13.8], [EY_B + 0.28, EY_B + 0.28],
        color=C_PRO, lw=1.4, ls="--", zorder=5)
ax.annotate("", xy=(5.10, EY_B + 0.33), xytext=(5.10, EY_B + 0.28),
            arrowprops=dict(arrowstyle="->", color=C_PRO, lw=1.4,
                            mutation_scale=8), zorder=6)
ax.plot([13.8, 13.8], [Y3 + 0.36, EY_B + 0.28],
        color=C_PRO, lw=1.4, ls="--", zorder=5)
note(ax, 14.5, (Y3 + 0.36 + EY_B - 0.33)/2 + 0.1,
     "prop_emb to\nQuery Proj", color=C_PRO, fs=6.3)

# Teacher prop note
note(ax, 1.8, Y3 - 0.65,
     "Teacher: plain MLP(48D→256→128)  (base_lin_vel included)",
     color=C_NOTE, fs=6.5)

# ─────────────────────────────────────────────────────────────────────────────
# BAND 4 – FUSION + DECODER   y_centre ≈ 3.45
# ─────────────────────────────────────────────────────────────────────────────
Y4 = 3.45
bg_box(ax, 0.3, 2.70, 17.5, 4.25, C_DEC, "Decoder")

rbox(ax,  4.5, Y4, 2.2, 0.68, C_DEC, C_DEC, "cat[ map ‖ prop ]",
     "192D + 128D = 320D")
rbox(ax,  8.5, Y4, 2.8, 0.68, C_DEC, C_DEC, "Decoder MLP",
     "Linear(320→512→256→12)\n@ 50 Hz", bold=True)
rbox(ax, 12.8, Y4, 2.0, 0.68, C_DEC, C_DEC, "Actions",
     "Joint PD targets\n12D", bold=True)

harrow(ax, 5.60, 7.10, Y4, color=C_DEC, label="320D")
harrow(ax, 9.90, 11.80, Y4, color=C_DEC, label="12D")

# map_emb drops from encoder to cat
lconnect(ax, 14.8, EY_B - 0.33, 4.5 - 0.80, Y4 + 0.34,
         color=C_ENC, label="map_emb 192D")
# prop_emb drops from prop row to cat
lconnect(ax, 13.8, Y3 - 0.36, 4.5 + 0.50, Y4 + 0.34,
         color=C_PRO, label="prop_emb 128D")

# Phase 1 & 2 labels
phase_y = Y4 + 0.10
note(ax, 17.9, Y4 + 0.28, "Phase 1", color=C_PRO, fs=7.5, style="normal")
note(ax, 17.9, Y4 + 0.00, "Teacher PPO  80k", color=C_PRO, fs=6.3)
note(ax, 17.9, Y4 - 0.25, "Phase 2", color=C_DEC, fs=7.5, style="normal")
note(ax, 17.9, Y4 - 0.50, "Student  40k iter", color=C_DEC, fs=6.3)


# ─────────────────────────────────────────────────────────────────────────────
# BAND 5 – ASYMMETRIC CRITIC (training only)
# ─────────────────────────────────────────────────────────────────────────────
Y5 = 1.65
bg_box(ax, 0.3, 0.75, 17.5, 2.55, C_CRIT,
       "Asymmetric Critic  (training only — same design for Teacher & Student)")

rbox(ax,  1.8, Y5, 2.0, 0.65, C_CRIT, C_CRIT, "Critic Prop",
     "50D = base_vel(3)\n+hist(42)+cmd_critic(5)")
rbox(ax,  4.6, Y5, 2.2, 0.65, C_CRIT, C_CRIT, "TeacherPropEnc",
     "MLP(50→256→128)\nprop_emb  128D")
rbox(ax,  7.5, Y5, 2.0, 0.65, C_CRIT, C_CRIT, "GT Map (3ch)",
     "RayCaster 14×36\n(x,y,z) per cell")
rbox(ax, 10.4, Y5, 2.3, 0.65, C_CRIT, C_CRIT, "AME2Encoder",
     "same arch, sep. weights\nmap_emb  192D")
rbox(ax, 13.1, Y5, 2.0, 0.65, C_CRIT, C_CRIT, "Contact (4D)",
     "per-foot\ncontact state")
rbox(ax, 16.0, Y5, 2.3, 0.70, C_CRIT, C_CRIT, "MoE  (4 experts)",
     "gate(contact) → weights\nΣ gate_i · expert_i\n→ V(state)", bold=True)

harrow(ax,  2.8,  3.50, Y5, color=C_CRIT)
harrow(ax,  5.70, 6.50, Y5, color=C_CRIT)
harrow(ax,  8.50, 9.25, Y5, color=C_CRIT)
harrow(ax, 11.55, 12.10, Y5, color=C_CRIT)
harrow(ax, 14.10, 14.85, Y5, color=C_CRIT)

note(ax, 10.4, Y5 - 0.58,
     "L-R symmetry augmentation:  V = 0.5 · (V_orig + V_flip)   "
     "— critic only, not actor  (Sec.IV-B)",
     color=C_CRIT, fs=6.5)

# ─────────────────────────────────────────────────────────────────────────────
# Map input drop  (top of band 2, from Band 1 to Band 2)
# ─────────────────────────────────────────────────────────────────────────────
# Policy Map Input box → drops into encoder
varrow(ax, 15.3, 9.35, EY_A + 0.33, color=C_MAP, label=" ", label_ha="left")
ax.plot([1.6, 15.3], [EY_A + 0.33, EY_A + 0.33],
        color=C_MAP, lw=1.4, zorder=5)
ax.annotate("", xy=(2.0 - 1.05/2, EY_A + 0.33), xytext=(1.6, EY_A + 0.33),
            arrowprops=dict(arrowstyle="->", color=C_MAP, lw=1.4,
                            mutation_scale=9), zorder=6)
ax.annotate("", xy=(4.8 - 1.0, EY_A + 0.33), xytext=(1.6, EY_A + 0.33),
            arrowprops=dict(arrowstyle="->", color=C_MAP, lw=1.4,
                            mutation_scale=9), zorder=6)

# ─────────────────────────────────────────────────────────────────────────────
# LEGEND
# ─────────────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(color=C_MAP,  label="Perception / Mapping"),
    mpatches.Patch(color=C_PRO,  label="Proprioception  (student: LSIO; teacher: MLP)"),
    mpatches.Patch(color=C_ENC,  label="AME-2 Encoder  (CNN + Pos Emb + MHA)"),
    mpatches.Patch(color=C_DEC,  label="Decoder / Actions"),
    mpatches.Patch(color=C_CRIT, label="Asymmetric MoE Critic  (training only)"),
]
ax.legend(handles=handles, loc="lower center",
          bbox_to_anchor=(0.50, -0.01),
          ncol=3, fontsize=7.5,
          framealpha=0.85, edgecolor="#CCCCCC")

# ─────────────────────────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0.03, 1, 1])
out = "docs/architecture.png"
plt.savefig(out, dpi=180, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
print(f"Saved → {out}")
