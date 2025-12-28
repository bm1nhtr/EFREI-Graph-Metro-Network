#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projet Graph - Réseau Métro
Point d'entrée principal du projet
"""

import sys
import os
import numpy as np

from algorithms.utils import load_graph_data

# Ajouter le dossier parent au path pour permettre les imports relatifs
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import des modules depuis le package algorithms
from algorithms import graph_network, visualize_metro
from algorithms.bfs import BFS

START_NODE = 20  # Noeud de départ pour TOUS les ALGOS


def main():
    """Fonction principale"""
    print("=" * 70)
    print("PROJET GRAPH - RESEAU METRO")
    print("=" * 70)
    print("\nChoisissez une option :")
    print("1. Generer le graphe (si pas encore fait)")
    print("2. Visualiser le reseau metro")
    print("3. BFS sur le reseau metro")
    print("0. Quitter")

    # Charger les données du graphe depuis un fichier numpy
    graph_data = load_graph_data('metro_network.npy') # <class 'numpy.ndarray'>   
    
    choice = input("\nVotre choix (0-3): ").strip()
    
    if choice == "1":
        print("\nGeneration du graphe...")
        # Le graphe est deja genere dans graph_network.py
        # On peut juste verifier qu'il existe
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        npy_file = os.path.join(data_dir, 'metro_network.npy')
        if os.path.exists(npy_file):
            print("Le graphe existe deja dans data/metro_network.npy")
        else:
            print("Generation en cours...")
            # Re-executer le script de generation
            algorithms_path = os.path.join(project_root, 'algorithms')
            exec(open(os.path.join(algorithms_path, 'graph_network.py')).read())
            print("Graphe genere avec succes!")
    
    elif choice == "2":
        print("\nVisualisation du reseau metro...")
        visualize_metro.visualize_metro_network()
        print("\nVisualisation sauvegardee dans results/metro_network_visualization.png")
    
    elif choice == "3":
        bfs = BFS(graph_data)
        print("\nExecution de l'algorithme BFS sur le reseau metro...")
        print(f'[Verification] Le graphe a {bfs.graph_data.shape[0]} connexions et {bfs.graph_data.shape[1]} colonnes') # (bfs.graph_data.shape)  
        parcours = bfs.parcourir_bfs(start_node= START_NODE)
        print("\nOrdre des stations visitees par BFS a partir de la station 0   :")
        print(len(parcours), "stations visitees.")
        print(parcours)
        bfs.sauvegarder_resultats(parcours, file_name='bfs_result.txt')
        print("\nResultats BFS sauvegardes dans results/bfs_result.txt")
        bfs.visualiser_parcours(parcours, start_node=START_NODE, file_name='bfs_visualization.png')
        print("Visualisation du parcours BFS sauvegardee dans results/bfs_visualization.png")
        bfs.visualiser_arbre_bfs(parcours, start_node=START_NODE, file_name='bfs_tree_visualization.png')
        print("Visualisation de l'arbre BFS sauvegardee dans results/bfs_tree_visualization.png")

        sys.exit(0)
    
    elif choice == "0":
        print("\nAu revoir!")
        sys.exit(0)
    
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
