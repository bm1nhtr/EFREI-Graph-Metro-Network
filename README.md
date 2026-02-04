# Réseau Métro 

## Auteurs
Binh Minh TRAN - Marouane NOUARA

## Description
Ce projet implémente un algorithme de recherche du plus court chemin dans un graphe non orienté représentant un réseau de métro. Le réseau contient 19 stations (0-18) réparties sur 3 lignes avec des stations de correspondance.

## Structure du Projet
```
ProjetGraph/
├── README.md                    # Ce fichier
├── main.py                      # Point d'entrée principal (menu console)
├── requirements.txt             # Dépendances Python
├── data/                        # Données du graphe
│   ├── metro_network.npy        # Format binaire NumPy
│   └── metro_network.txt        # Format texte lisible
├── algorithms/                  # Implémentations des algorithmes
│   ├── __init__.py
│   ├── graph_network.py         # Génération du graphe
│   ├── visualize_metro.py       # Visualisation du réseau
│   ├── utils.py                 # Utilitaires (chargement graphe, layouts)
│   ├── bfs.py                   # BFS (parcours + arbre)
│   ├── prim.py                  # Prim (MST)
│   └── bellman_ford.py          # Bellman-Ford (plus courts chemins)
├── interface/                   # Interfaces utilisateur (web + desktop)
│   ├── app.py                   # Serveur web Flask
│   ├── gui.py                   # Interface graphique desktop (Tkinter)
│   └── templates/
│       └── index.html           # Page web (visualisations)
└── results/                     # Résultats et visualisations
    ├── BFS/, PRIM/, BELLMAN_FORD/
    ├── metro_network_visualization.png
    └── REFERENCE_GRAPHE_VERIFICATION.txt
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

### Interface graphique (bonus)
```bash
python interface/gui.py
```
Ou choisir l’option **6** dans le menu de `main.py`.  
La GUI permet d’afficher le graphe de façon dynamique (zoom, pan), de choisir la station de départ (0–18) et de lancer chaque algorithme (BFS, Prim, Bellman-Ford) avec visualisation en couleur.

### Serveur Web (accès depuis l’extérieur)
```bash
pip install -r requirements.txt
python interface/app.py
```
(Le fichier `requirements.txt` inclut `flask`.)
**Accès en local :**
- **Sur la même machine :** http://localhost:5000  
- **Depuis un autre appareil (réseau local) :** http://\<IP_DE_LA_MACHINE\>:5000  

La page permet de choisir la visualisation (graphe métro, BFS, Prim, Bellman-Ford) et la station de départ, puis d’afficher l’image générée. Pour exposer sur internet, déployer sur un hébergeur (ex. Render, Railway, Heroku) ou ouvrir le port 5000 sur la box et utiliser l’IP publique.

## Notes
- Le graphe est **non orienté** (undirected), donc il existe toujours un chemin entre deux stations quelconques.
- Les poids des arêtes représentent le temps de trajet en minutes (1-4 minutes).
- Les fichiers de visualisation et d'implémentation de l'algorithme doivent être créés par les étudiants.
