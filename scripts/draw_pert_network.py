#!/usr/bin/env python3
"""
Génère l'image du réseau PERT (équivalent du TikZ dans le rapport).
Réseau de tâches A–J avec durées, et flèches de dépendances.

Usage (depuis la racine du projet) :
    python scripts/draw_pert_network.py
"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Racine du projet
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Sortie : results/ ou dossier du script
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.normpath(os.path.join(OUTPUT_DIR, "pert_network.png"))


def main():
    # Positions (x, y) — layout comme dans le TikZ (A gauche, B à droite de A, C en dessous de B, etc.)
    pos = {
        "A": (0.0, 0.6),
        "B": (1.2, 0.6),
        "C": (1.2, 0.0),
        "D": (2.4, 0.6),
        "E": (3.6, 0.6),
        "F": (4.8, 0.6),
        "G": (4.8, 0.0),
        "H": (6.0, 0.6),
        "I": (7.2, 0.6),
        "J": (8.4, 0.6),
    }
    durations = {
        "A": 2, "B": 3, "C": 2, "D": 2, "E": 3, "F": 4, "G": 2, "H": 2, "I": 1, "J": 1,
    }
    edges = [
        ("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E"),
        ("E", "F"), ("E", "G"), ("F", "H"), ("G", "I"), ("H", "I"), ("I", "J"),
    ]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_aspect("equal")
    ax.axis("off")

    box_w, box_h = 0.35, 0.28

    # Boîtes arrondies (task) + labels
    for name, (x, y) in pos.items():
        d = durations[name]
        box = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="white",
            edgecolor="black",
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(x, y + 0.04, name, ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(x, y - 0.06, str(d), ha="center", va="center", fontsize=9, color="gray")

    # Flèches : bord de la boîte source -> bord de la boîte cible
    for u, v in edges:
        xu, yu = pos[u]
        xv, yv = pos[v]
        dx, dy = xv - xu, yv - yu
        dist = (dx * dx + dy * dy) ** 0.5 or 1.0
        # Décalage le long de la direction pour partir/arriver au bord
        offset = 0.5 * (box_w if abs(dx) > abs(dy) else box_h)
        x1 = xu + (dx / dist) * offset
        y1 = yu + (dy / dist) * offset
        x2 = xv - (dx / dist) * offset
        y2 = yv - (dy / dist) * offset
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=2, color="black"),
        )

    ax.set_xlim(-0.5, 9)
    ax.set_ylim(-0.35, 0.95)
    ax.set_title("Réseau PERT du projet (durées indiquées dans chaque tâche)", fontsize=11, pad=10)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_PATH,
        dpi=100,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compress_level": 6},
    )
    plt.close()
    print(f"[OK] Image sauvegardée : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
