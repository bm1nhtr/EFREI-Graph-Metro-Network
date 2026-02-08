#!/usr/bin/env python3
"""
Projet Graph - Réseau Métro.

Point d'entrée principal : menu console pour lancer les algorithmes
(BFS, DFS, Prim, Kruskal, Dijkstra, Bellman-Ford, Floyd-Warshall)
et les visualisations. Les résultats et images sont sauvegardés dans results/.
"""

import os
import subprocess
import sys

from algorithms import visualize_metro
from algorithms.bellman_ford import BellmanFord
from algorithms.bfs import BFS
from algorithms.dfs import DFS
from algorithms.dijkstra import Dijkstra
from algorithms.floyd_warshall import FloydWarshall
from algorithms.kruskal import Kruskal
from algorithms.prim import Prim
from algorithms.utils import load_graph_data

# Racine du projet (pour sous-process et chemins)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Noeud de départ par défaut pour les algorithmes (0-18, 19 stations)
DEFAULT_START_NODE = 10
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
NPY_FILE = os.path.join(DATA_DIR, "metro_network.npy")


def _ensure_graph_exists() -> bool:
    """Vérifie que data/metro_network.npy existe ; sinon le génère. Retourne True si OK."""
    if os.path.exists(NPY_FILE):
        return True
    print("Generation en cours...")
    script_path = os.path.join(PROJECT_ROOT, "algorithms", "graph_network.py")
    subprocess.run([sys.executable, script_path], cwd=PROJECT_ROOT, check=True)
    return True


def _format_distance(d: float) -> str:
    """Retourne la distance sous forme lisible (entier ou 'inf')."""
    return str(int(d)) if d != float("inf") else "inf"


def _run_bfs(graph_data, start_node: int) -> None:
    """Lance BFS (parcours en largeur), sauvegarde résultats et visualisations dans results/BFS/."""
    print("Execution de l'algorithme BFS sur le reseau metro...")
    bfs = BFS(graph_data)
    parcours = bfs.parcourir_bfs(start_node=start_node)
    print("\nOrdre des stations visitees par BFS :")
    print(len(parcours), "stations visitees.")
    print(parcours)
    bfs.sauvegarder_resultats(parcours, file_name="bfs_result.txt")
    bfs.visualiser_parcours(parcours, start_node=start_node, file_name="bfs_visualization.png")
    bfs.visualiser_arbre_bfs(
        parcours, start_node=start_node, file_name="bfs_tree_visualization.png"
    )
    print("Resultats et visualisations BFS sauvegardes dans results/BFS/")


def _run_dfs(graph_data, start_node: int) -> None:
    """Lance DFS (parcours en profondeur), sauvegarde résultats et visualisations dans results/DFS/."""
    print("Execution de l'algorithme DFS sur le reseau metro...")
    dfs = DFS(graph_data)
    parcours, parent = dfs.parcourir_dfs(start_node=start_node)
    print("\nOrdre des stations visitees par DFS :")
    print(len(parcours), "stations visitees.")
    print(parcours)
    dfs.sauvegarder_resultats(parcours, file_name="dfs_result.txt")
    dfs.visualiser_parcours(
        parcours, parent, start_node=start_node, file_name="dfs_visualization.png"
    )
    dfs.visualiser_arbre_dfs(
        parcours, parent, start_node=start_node, file_name="dfs_tree_visualization.png"
    )
    print("Resultats et visualisations DFS sauvegardes dans results/DFS/")


def _run_prim(graph_data, start_node: int) -> None:
    """Lance Prim (MST - arbre couvrant minimum), sauvegarde résultats et visuels dans results/PRIM/."""
    print("Execution de l'algorithme de Prim (MST) sur le reseau metro...")
    prim = Prim(graph_data)
    mst_edges, total_weight = prim.prim_mst(start_node=start_node)
    print(f"\nMST: {len(mst_edges)} aretes, poids total = {total_weight}")
    prim.sauvegarder_resultats(mst_edges, total_weight, file_name="prim_result.txt")
    prim.visualiser_mst(
        mst_edges, total_weight, start_node=start_node, file_name="prim_visualization.png"
    )
    prim.visualiser_arbre_mst(
        mst_edges, total_weight, start_node=start_node, file_name="prim_tree_visualization.png"
    )
    print("Resultats et visualisations Prim sauvegardes dans results/PRIM/")


def _run_kruskal(graph_data, start_node: int) -> None:
    """Lance Kruskal (MST - arbre couvrant minimum), sauvegarde résultats et visuels dans results/KRUSKAL/."""
    print("Execution de l'algorithme de Kruskal (MST) sur le reseau metro...")
    kr = Kruskal(graph_data)
    mst_edges, total_weight = kr.kruskal_mst(start_node=start_node)
    print(f"\nMST: {len(mst_edges)} aretes, poids total = {total_weight}")
    kr.sauvegarder_resultats(mst_edges, total_weight, file_name="kruskal_result.txt")
    kr.visualiser_mst(
        mst_edges, total_weight, start_node=start_node, file_name="kruskal_visualization.png"
    )
    kr.visualiser_arbre_mst(
        mst_edges, total_weight, start_node=start_node, file_name="kruskal_tree_visualization.png"
    )
    print("Resultats et visualisations Kruskal sauvegardes dans results/KRUSKAL/")


def _run_bellman_ford(graph_data, start_node: int) -> None:
    """Lance Bellman-Ford (PCC avec détection cycle négatif), graphe option A, sauvegarde dans results/BELLMAN_FORD/."""
    try:
        graph_data = load_graph_data("metro_network_bellman.npy")
    except FileNotFoundError:
        print(
            "Fichier data/metro_network_bellman.npy introuvable. Choisissez l'option 1 pour generer les graphes."
        )
        return
    print("Execution de l'algorithme Bellman-Ford (graphe cas controle, 1 poids negatif)...")
    bf = BellmanFord(graph_data)
    distances, predecessors, has_neg = bf.bellman_ford(start_node=start_node)
    if has_neg:
        print("[ATTENTION] Cycle de poids negatif detecte.")
    else:
        print(f"\nPlus courts chemins depuis la station {start_node}:")
        for node in sorted(distances.keys()):
            print(f"  Station {node}: distance = {_format_distance(distances[node])}")
    bf.sauvegarder_resultats(
        distances, predecessors, start_node, has_neg, file_name="bellman_ford_result.txt"
    )
    bf.visualiser_parcours(
        distances, predecessors, start_node, has_neg, file_name="bellman_ford_visualization.png"
    )
    print("Resultats et visualisation Bellman-Ford sauvegardes dans results/BELLMAN_FORD/")


def _run_dijkstra(graph_data, start_node: int) -> None:
    """Lance Dijkstra (plus courts chemins depuis une source), sauvegarde résultats dans results/DIJKSTRA/."""
    print("Execution de l'algorithme de Dijkstra sur le reseau metro...")
    dj = Dijkstra(graph_data)
    distances, predecessors = dj.dijkstra(start_node=start_node)
    print(f"\nPlus courts chemins depuis la station {start_node}:")
    for node in sorted(distances.keys()):
        print(f"  Station {node}: distance = {_format_distance(distances[node])}")
    dj.sauvegarder_resultats(distances, predecessors, start_node, file_name="dijkstra_result.txt")
    dj.visualiser_parcours(
        distances, predecessors, start_node, file_name="dijkstra_visualization.png"
    )
    print("Resultats et visualisation Dijkstra sauvegardes dans results/DIJKSTRA/")


def _run_floyd_warshall(graph_data) -> None:
    """Lance Floyd-Warshall (plus courts chemins toutes paires + centralité), sauvegarde dans results/FLOYD_WARSHALL/."""
    print("Execution de l'algorithme de Floyd-Warshall (toutes les paires)...")
    fw = FloydWarshall(graph_data)
    dist = fw.floyd_warshall()
    fw.sauvegarder_resultats(dist, file_name="floyd_warshall_result.txt")
    fw.visualiser_matrice(dist, file_name="floyd_warshall_visualization.png")
    print("Resultats et visualisation Floyd-Warshall sauvegardes dans results/FLOYD_WARSHALL/")


def _run_gui() -> None:
    """Ouvre l'interface graphique desktop (Tkinter)."""
    from interface.gui import MetroGraphApp

    app = MetroGraphApp()
    app.run()


def main() -> None:
    """Boucle principale : demande la station de départ, affiche le menu et exécute l'option jusqu'à Quitter (0)."""
    start_node = DEFAULT_START_NODE
    # Station de départ : utilisée par Bellman-Ford, Dijkstra, Prim, BFS, DFS (Kruskal n'en a pas besoin)
    raw = input(
        "Station de depart pour options 3, 4, 5, 7 et 8 (0-18). Entree = 10 (defaut) : "
    ).strip()
    if raw != "":
        try:
            n = int(raw)
            if 0 <= n <= 18:
                start_node = n
            else:
                print("Valeur invalide : defaut 10 utilise.")
        except ValueError:
            print("Valeur invalide : defaut 10 utilise.")

    while True:
        print("\n" + "=" * 70)
        print("PROJET GRAPH - RESEAU METRO")
        print("=" * 70)
        print("\nChoisissez une option :")
        print("  1. Generer le graphe (si pas encore fait)")
        print("  2. Visualiser le reseau metro")
        print("  3. Bellman-Ford (PCC)")
        print("  4. Dijkstra (PCC)")
        print("  5. Prim (MST)")
        print("  6. Kruskal (MST)")
        print("  7. BFS sur le reseau metro")
        print("  8. DFS sur le reseau metro")
        print("  9. Floyd-Warshall (toutes les paires + analyse centrale)")
        print(" 10. Interface graphique (GUI desktop)")
        print("  0. Quitter")
        choice = input("\nVotre choix (0-10): ").strip()

        if choice == "0":
            print("\nAu revoir!")
            break

        if choice == "1":
            print("\nGeneration du graphe...")
            if os.path.exists(NPY_FILE):
                print("Le graphe existe deja dans data/metro_network.npy")
            else:
                _ensure_graph_exists()
                print("Graphe genere avec succes!")
            continue

        try:
            graph_data = load_graph_data("metro_network.npy")
        except FileNotFoundError:
            print("\nFichier data/metro_network.npy introuvable. Choisissez d'abord l'option 1.")
            continue

        handlers = {
            "3": lambda: _run_bellman_ford(graph_data, start_node),
            "4": lambda: _run_dijkstra(graph_data, start_node),
            "5": lambda: _run_prim(graph_data, start_node),
            "6": lambda: _run_kruskal(graph_data, start_node),
            "7": lambda: _run_bfs(graph_data, start_node),
            "8": lambda: _run_dfs(graph_data, start_node),
            "9": lambda: _run_floyd_warshall(graph_data),
            "10": _run_gui,
        }
        if choice == "2":
            print("\nVisualisation du reseau metro...")
            visualize_metro.visualize_metro_network()
            try:
                load_graph_data("metro_network_bellman.npy")
                visualize_metro.visualize_metro_network(
                    use_bellman_graph=True,
                    output_filename="metro_network_bellman_visualization.png",
                )
                print(
                    "Graphe cas controle (Bellman, 1 poids negatif) : "
                    "results/metro_network_bellman_visualization.png"
                )
            except FileNotFoundError:
                pass
            continue
        if choice in handlers:
            handlers[choice]()
        else:
            print("\nChoix invalide!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterruption par l'utilisateur. Au revoir!")
        sys.exit(0)
    except Exception as e:
        print(f"\nErreur: {e}")
        sys.exit(1)
