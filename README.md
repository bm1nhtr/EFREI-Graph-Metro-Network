# [EFREI] Projet Graph - Réseau Métro

## Auteurs
Binh Minh TRAN - Marouane NOUARA

## Description
Ce projet implémente un algorithme de recherche du plus court chemin dans un graphe non orienté représentant un réseau de métro. Le réseau contient 28 stations réparties sur 3 lignes avec des stations de correspondance.

## Structure du Projet
```
ProjetGraph/
├── README.md                 # Ce fichier
├── main.py                   # Point d'entrée principal
├── requirements.txt           # Dépendances Python
├── data/                     # Données du graphe
│   ├── metro_network.npy    # Format binaire NumPy
│   └── metro_network.txt     # Format texte lisible
├── algorithms/               # Implémentations des algorithmes
│   ├── graph_network.py     # Génération du graphe
│   └── visualize_metro.py   # Visualisation du réseau
└── results/                  # Résultats et visualisations
    └── *.png                 
```

## Prérequis
- Python 3.7 ou supérieur
- Bibliothèques Python :
  - numpy
  - matplotlib
  - networkx

## Installation

1. Installer les dépendances :
```bash
pip install -r requirements.txt
```

2. Générer les données du graphe :
```bash
python algorithms/graph_network.py
```

## Utilisation

### Exécution principale
```bash
python main.py
```

## Notes
- Le graphe est **non orienté** (undirected), donc il existe toujours un chemin entre deux stations quelconques.
- Les poids des arêtes représentent le temps de trajet en minutes (1-4 minutes).
- Les fichiers de visualisation et d'implémentation de l'algorithme doivent être créés par les étudiants.
