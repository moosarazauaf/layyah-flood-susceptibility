"""
Policy figures, one file each at 600 dpi, 6.5 in wide, no text under 9.5 pt.
Inputs: outputs/layyah_map_stats.json, outputs/population_growth.json.

figD_flooding_by_class.png   share of observed flooded land in each class
figE_population_growth.png   population 2000/2010/2020 on class 4-5 land,
                             against the corridor and the whole tehsil
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "outputs")
FIG = os.path.join(_ROOT, "paper", "figures")
DPI, W, FS = 600, 6.5, 10
CLASS_COL = ["#ffffb2", "#fecc5c", "#fd8d3c", "#e31a1c", "#800026"]


def flooding_by_class(st):
    sets = [("flooded_2000_2019", "Flooded 2000-2019"),
            ("first_flooded_2020_2021", "First flooded 2020-2021"),
            ("all", "All corridor land (reference)")]
    fig, ax = plt.subplots(figsize=(W, 3.4), layout="constrained")
    for i, (key, lab) in enumerate(sets):
        v = st["area_km2"][key]
        tot = sum(v.values())
        left = 0
        for c in range(1, 6):
            share = 100 * v.get(str(c), 0) / tot
            ax.barh(i, share, left=left, color=CLASS_COL[c - 1],
                    edgecolor="white", linewidth=0.6)
            if share >= 7:
                ax.text(left + share / 2, i, f"{share:.0f}%", ha="center", va="center",
                        fontsize=9.5, color="white" if c >= 4 else "black")
            left += share
    ax.set_yticks(range(len(sets)), [s[1] for s in sets], fontsize=FS)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of land in each susceptibility class (%)", fontsize=11)
    ax.tick_params(labelsize=FS)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLASS_COL]
    ax.legend(handles, ["1 (lowest)", "2", "3", "4", "5 (highest)"], ncol=5, frameon=False,
              fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(FIG, "figD_flooding_by_class.png"), dpi=DPI, facecolor="white")
    plt.close(fig); print("wrote figD_flooding_by_class.png")


def population_growth(gr):
    teh = ["Leiah", "Karor Lal Esan"]
    groups = [("tehsil", "Whole tehsil", "#bdbdbd"), ("corridor", "Within 15 km of the Indus", "#6baed6"),
              ("classes_4_5", "Susceptibility classes 4-5", "#800026")]
    fig, ax = plt.subplots(figsize=(W, 3.6), layout="constrained")
    x = np.arange(len(teh))
    for j, (key, lab, col) in enumerate(groups):
        vals = [gr[t]["ratio_2020_2000"][key] for t in teh]
        ax.bar(x + (j - 1) * 0.26, vals, 0.26, color=col, label=lab)
        for xi, v in zip(x + (j - 1) * 0.26, vals):
            ax.text(xi, v + 0.04, f"×{v:.2f}", ha="center", fontsize=9.5)
    ax.axhline(1, color="black", lw=0.8)
    ax.set_xticks(x, teh, fontsize=FS)
    ax.set_ylabel("Population 2020 ÷ population 2000", fontsize=11)
    ax.set_ylim(0, max(gr[t]["ratio_2020_2000"]["classes_4_5"] for t in teh) * 1.2)
    ax.tick_params(labelsize=FS)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(FIG, "figE_population_growth.png"), dpi=DPI, facecolor="white")
    plt.close(fig); print("wrote figE_population_growth.png")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    flooding_by_class(json.load(open(os.path.join(OUT, "layyah_map_stats.json"), encoding="utf-8")))
    population_growth(json.load(open(os.path.join(OUT, "population_growth.json"), encoding="utf-8")))
