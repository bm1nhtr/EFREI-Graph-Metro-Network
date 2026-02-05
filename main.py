#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projet Graph - Réseau Métro
Point d'entrée principal du projet
"""

import sys
import os
import subprocess

from algorithms.utils import load_graph_data

# Ajouter le dossier parent au path pour permettre les imports relatifs
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import des modules depuis le package algorithms
from algorithms import graph_network, visualize_metro
from algorithms.bfs import BFS
from algorithms.prim import Prim
from algorithms.bellman_ford import BellmanFord
from algorithms.kruskal import Kruskal
from algorithms.dfs import DFS
from algorithms.dijkstra import Dijkstra


START_NODE = 10  # Noeud de départ pour TOUS les ALGOS (0-18, 19 stations)


def main():
    """Fonction principale — menu en boucle jusqu'à ce que l'utilisateur choisisse 0 (Quitter)."""
    while True:
        print("=" * 70)
        print("PROJET GRAPH - RESEAU METRO")
        print("=" * 70)
        print("\nChoisissez une option :")
        print("1. Generer le graphe (si pas encore fait)")
        print("2. Visualiser le reseau metro")
        print("3. BFS sur le reseau metro")
        print("4. Prim (MST - Arbre couvrant minimum)")
        print("5. Bellman-Ford (Plus courts chemins)")
        print("6. Kruskal (MST - Arbre couvrant minimum)")
        print("7. DFS sur le reseau metro")
        print("8. Dijkstra (Plus courts chemins)")
        print("9. Interface graphique (GUI desktop)")
        print("0. Quitter")

        choice = input("\nVotre choix (0-9): ").strip()

        if choice == "0":
            print("\nAu revoir!")
            break

        if choice == "1":
            print("\nGeneration du graphe...")
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            npy_file = os.path.join(data_dir, 'metro_network.npy')
            if os.path.exists(npy_file):
                print("Le graphe existe deja dans data/metro_network.npy")
            else:
                print("Generation en cours...")
                script_path = os.path.join(project_root, 'algorithms', 'graph_network.py')
                subprocess.run([sys.executable, script_path], cwd=project_root, check=True)
                print("Graphe genere avec succes!")
            continue

        try:
            graph_data = load_graph_data('metro_network.npy')
        except FileNotFoundError:
            print("\nFichier data/metro_network.npy introuvable. Choisissez d'abord l'option 1.")
            continue

        if choice == "2":
            print("\nVisualisation du reseau metro...")
            visualize_metro.visualize_metro_network()
            print("\nVisualisation sauvegardee dans results/metro_network_visualization.png")

        elif choice == "3":
            bfs = BFS(graph_data)
            print("\nExecution de l'algorithme BFS sur le reseau metro...")
            print(f'[Verification] Le graphe a {bfs.graph_data.shape[0]} connexions et {bfs.graph_data.shape[1]} colonnes')
            parcours = bfs.parcourir_bfs(start_node=START_NODE)
            print("\nOrdre des stations visitees par BFS :")
            print(len(parcours), "stations visitees.")
            print(parcours)
            bfs.sauvegarder_resultats(parcours, file_name='bfs_result.txt')
            bfs.visualiser_parcours(parcours, start_node=START_NODE, file_name='bfs_visualization.png')
            bfs.visualiser_arbre_bfs(parcours, start_node=START_NODE, file_name='bfs_tree_visualization.png')
            print("Resultats et visualisations BFS sauvegardes dans results/BFS/")

        elif choice == "4":
            prim = Prim(graph_data)
            print("\nExecution de l'algorithme de Prim (MST) sur le reseau metro...")
            mst_edges, total_weight = prim.prim_mst(start_node=START_NODE)
            print(f"\nMST: {len(mst_edges)} aretes, poids total = {total_weight}")
            prim.sauvegarder_resultats(mst_edges, total_weight, file_name='prim_result.txt')
            prim.visualiser_mst(mst_edges, total_weight, start_node=START_NODE, file_name='prim_visualization.png')
            print("Resultats et visualisation Prim sauvegardes dans results/PRIM/")

        elif choice == "5":
            bf = BellmanFord(graph_data)
            print("\nExecution de l'algorithme Bellman-Ford sur le reseau metro...")
            distances, predecessors, has_neg = bf.bellman_ford(start_node=START_NODE)
            if has_neg:
                print("[ATTENTION] Cycle de poids negatif detecte.")
            else:
                print(f"\nPlus courts chemins depuis la station {START_NODE}:")
                for node in sorted(distances.keys()):
                    d = distances[node]
                    print(f"  Station {node}: distance = {int(d) if d != float('inf') else 'inf'}")
            bf.sauvegarder_resultats(distances, predecessors, START_NODE, has_neg, file_name='bellman_ford_result.txt')
            bf.visualiser_parcours(distances, predecessors, START_NODE, has_neg, file_name='bellman_ford_visualization.png')
            print("Resultats et visualisation Bellman-Ford sauvegardes dans results/BELLMAN_FORD/")
    

        elif choice == "6":
            kr = Kruskal(graph_data)
            print("\nExecution de l'algorithme de Kruskal (MST) sur le reseau metro...")
            mst_edges, total_weight = kr.kruskal_mst(start_node=START_NODE)
            print(f"\nMST: {len(mst_edges)} aretes, poids total = {total_weight}")
            kr.sauvegarder_resultats(mst_edges, total_weight, file_name='kruskal_result.txt')
            kr.visualiser_mst(mst_edges, total_weight, start_node=START_NODE, file_name='kruskal_visualization.png')
            print("Resultats et visualisation Kruskal sauvegardes dans results/KRUSKAL/")


        elif choice == "7":
            dfs = DFS(graph_data)
            print("\nExecution de l'algorithme DFS sur le reseau metro...")
            print(f'[Verification] Le graphe a {dfs.graph_data.shape[0]} connexions et {dfs.graph_data.shape[1]} colonnes')

            parcours, parent = dfs.parcourir_dfs(start_node=START_NODE)

            print("\nOrdre des stations visitees par DFS :")
            print(len(parcours), "stations visitees.")
            print(parcours)

            dfs.sauvegarder_resultats(parcours, file_name='dfs_result.txt')
            dfs.visualiser_parcours(parcours,parent, start_node=START_NODE, file_name='dfs_visualization.png')
            dfs.visualiser_arbre_dfs(parcours, parent, start_node=START_NODE, file_name='dfs_tree_visualization.png')

            print("Resultats et visualisations DFS sauvegardes dans results/DFS/")

        elif choice == "8":
            dj = Dijkstra(graph_data)
            print("\nExecution de l'algorithme de Dijkstra sur le reseau metro...")

            # Calcul des plus courts chemins
            distances, predecessors = dj.dijkstra(start_node=START_NODE)

            print(f"\nPlus courts chemins depuis la station {START_NODE}:")
            for node in sorted(distances.keys()):
                d = distances[node]
                print(f"  Station {node}: distance = {int(d) if d != float('inf') else 'inf'}")

            # Sauvegarde des résultats
            dj.sauvegarder_resultats(distances,predecessors,START_NODE,file_name="dijkstra_result.txt")

            # Visualisation
            dj.visualiser_parcours(distances,predecessors,START_NODE,file_name="dijkstra_visualization.png")

            print("Resultats et visualisation Dijkstra sauvegardes dans results/DIJKSTRA/")

        elif choice == "9":
            from interface.gui import MetroGraphApp
            app = MetroGraphApp()
            app.run()

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
