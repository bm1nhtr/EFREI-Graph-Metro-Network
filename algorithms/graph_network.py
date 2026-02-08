"""
Graphe Réseau Métro - Non orienté.

Représente un système de métro avec des stations et des connexions.
Nœuds : stations (0-18). Arêtes : connexions avec poids (temps en minutes).
Le graphe est NON ORIENTÉ.

Dataset : propre graphe thématique (réseau de transport), disponible en
JSON, CSV, TXT et NPY dans data/.
"""

import json
import os

import numpy as np

# Réseau Métro : 19 stations (0-18)
# Format : [station_a, station_b, temps_trajet_minutes]
# Chaque arête = connexion bidirectionnelle

METRO_NETWORK = np.array(
    [
        # Ligne 1 (7 stations : 0-6)
        [0, 1, 3],
        [1, 2, 2],
        [2, 3, 4],
        [3, 4, 3],
        [4, 5, 2],
        [5, 6, 3],
        # Ligne 2 (5 stations : 7-11)
        [7, 8, 3],
        [8, 9, 2],
        [9, 10, 4],
        [10, 11, 3],
        # Ligne 3 (7 stations : 12-18)
        [12, 13, 3],
        [13, 14, 2],
        [14, 15, 4],
        [15, 16, 3],
        [16, 17, 2],
        [17, 18, 3],
        # Correspondances
        [3, 8, 1],
        [5, 10, 1],
        [6, 13, 1],
        [4, 15, 1],
    ],
    dtype=int,
)

# Variante pour Bellman-Ford : cas contrôlé avec 1 poids négatif.
# Arête (3, 8) : 1 -> -1 pour créer un cycle négatif 3-8-3 (détection par Bellman-Ford).
# Les autres algorithmes utilisent METRO_NETWORK (sans poids négatif).
METRO_NETWORK_BELLMAN = METRO_NETWORK.copy()
for i in range(len(METRO_NETWORK_BELLMAN)):
    if int(METRO_NETWORK_BELLMAN[i, 0]) == 3 and int(METRO_NETWORK_BELLMAN[i, 1]) == 8:
        METRO_NETWORK_BELLMAN[i, 2] = -1
        break
METRO_NETWORK_BELLMAN = METRO_NETWORK_BELLMAN.astype(np.int64)


def save_metro_network() -> None:
    """Sauvegarde le graphe métro en .npy, .txt, .csv, .json et variante Bellman dans data/."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    # NPY (utilisé par l'app)
    npy_path = os.path.join(data_dir, "metro_network.npy")
    np.save(npy_path, METRO_NETWORK)

    # TXT (lisible)
    txt_path = os.path.join(data_dir, "metro_network.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("# Graphe Réseau Métro (Non orienté)\n")
        f.write("# Format : station_a station_b temps_trajet_minutes\n")
        f.write("# Total stations : 19 (0-18)\n")
        f.write(f"# Total connexions : {len(METRO_NETWORK)}\n\n")
        for edge in METRO_NETWORK:
            f.write(f"{edge[0]} {edge[1]} {edge[2]}\n")

    # CSV (dataset standard)
    csv_path = os.path.join(data_dir, "metro_network.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write("station_a,station_b,temps_minutes\n")
        for edge in METRO_NETWORK:
            f.write(f"{edge[0]},{edge[1]},{edge[2]}\n")

    # JSON (dataset standard)
    json_path = os.path.join(data_dir, "metro_network.json")
    edges = [
        {"station_a": int(u), "station_b": int(v), "temps_minutes": int(w)}
        for u, v, w in METRO_NETWORK
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"description": "Réseau métro (graphe non orienté)", "stations": 19, "edges": edges},
            f,
            indent=2,
        )

    # Bellman-Ford : graphe avec 1 poids négatif (option A - sujet)
    bellman_npy = os.path.join(data_dir, "metro_network_bellman.npy")
    np.save(bellman_npy, METRO_NETWORK_BELLMAN)
    bellman_txt = os.path.join(data_dir, "metro_network_bellman.txt")
    with open(bellman_txt, "w", encoding="utf-8") as f:
        f.write("# Graphe métro - Variante Bellman-Ford (cas contrôlé)\n")
        f.write("# 1 poids négatif : arête (3,8) = -1 → cycle négatif 3-8-3\n")
        f.write("# Utilisé uniquement pour Bellman-Ford (détection cycle négatif).\n")
        f.write("# Total stations : 19 (0-18)\n")
        f.write(f"# Total connexions : {len(METRO_NETWORK_BELLMAN)}\n\n")
        for edge in METRO_NETWORK_BELLMAN:
            f.write(f"{edge[0]} {edge[1]} {edge[2]}\n")

    print(
        f"Fichiers créés : {npy_path}, {txt_path}, {csv_path}, {json_path}, {bellman_npy}, {bellman_txt}"
    )


if __name__ == "__main__":
    save_metro_network()
    print("=" * 60)
    print("Graphe Réseau Métro Créé")
    print("=" * 60)
    print("Total stations (nœuds) : 19 (0-18)")
    print(f"Total connexions (arêtes) : {len(METRO_NETWORK)}")
    print("Type de graphe : NON ORIENTÉ")
    print("=" * 60)
