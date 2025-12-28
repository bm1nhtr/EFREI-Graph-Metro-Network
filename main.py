#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projet Graph - Réseau Métro
Point d'entrée principal du projet
"""

import sys
import os

# Ajouter le dossier algorithms au path
algorithms_path = os.path.join(os.path.dirname(__file__), 'algorithms')
sys.path.insert(0, algorithms_path)

# Import des modules
import graph_network
import visualize_metro

def main():
    """Fonction principale"""
    print("=" * 70)
    print("PROJET GRAPH - RESEAU METRO")
    print("=" * 70)
    print("\nChoisissez une option :")
    print("1. Generer le graphe (si pas encore fait)")
    print("2. Visualiser le reseau metro")
    print("3. Quitter")
    
    choice = input("\nVotre choix (1-3): ").strip()
    
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
            exec(open(os.path.join(algorithms_path, 'graph_network.py')).read())
            print("Graphe genere avec succes!")
    
    elif choice == "2":
        print("\nVisualisation du reseau metro...")
        visualize_metro.visualize_metro_network()
        print("\nVisualisation sauvegardee dans results/metro_network_visualization.png")
    
    elif choice == "3":
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
