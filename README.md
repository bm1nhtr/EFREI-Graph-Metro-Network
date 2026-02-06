# Réseau Métro

## Auteurs
Binh Minh TRAN - Marouane NOUARA

**Démo en ligne (Render) :** [https://graph-metro-network.onrender.com/](https://graph-metro-network.onrender.com/)

## Description
Ce projet implémente des algorithmes de graphes sur un **réseau de métro** modélisé en graphe non orienté : recherche du plus court chemin (Dijkstra, Bellman-Ford), parcours (BFS, DFS), arbre couvrant minimum (Prim, Kruskal). Le réseau contient 19 stations (0-18) réparties sur 3 lignes avec des stations de correspondance.

**Données :** nous utilisons **notre propre dataset** — graphe thématique (réseau de transport). Le graphe est construit manuellement et disponible en plusieurs formats dans `data/` : JSON, CSV, TXT et NPY.

**Deux graphes :**  
- **Graphe principal** (`metro_network.*`) : utilisé par BFS, DFS, Prim, Kruskal, Dijkstra, Floyd-Warshall (poids positifs).  
- **Graphe Bellman-Ford** (`metro_network_bellman.*`) : **cas contrôlé** avec **un poids négatif** (arête 3–8 : 1 → -1). Ce graphe contient un cycle négatif (3-8-3) ; Bellman-Ford est le seul algorithme à l’utiliser, pour démontrer la **détection des cycles négatifs** et les distances minimales (option A du sujet).

## Structure du Projet

```
ProjetGraph/
├── README.md                    # Ce fichier
├── main.py                      # Point d'entrée principal (menu console)
├── requirements.txt             # Dépendances Python
├── pyproject.toml               # Config lint / format (Ruff)
├── data/                        # Dataset du graphe (propre graphe thématique)
│   ├── metro_network.npy        # Graphe principal (tous les algos sauf Bellman)
│   ├── metro_network.txt        # Texte lisible
│   ├── metro_network.csv        # CSV (dataset standard)
│   ├── metro_network.json       # JSON (dataset standard)
│   ├── metro_network_bellman.npy # Variante 1 poids négatif (Bellman-Ford)
│   └── metro_network_bellman.txt # Lisible (cas contrôlé)
├── algorithms/                  # Implémentations des algorithmes
│   ├── __init__.py
│   ├── graph_network.py         # Génération du graphe (données + script)
│   ├── visualize_metro.py       # Visualisation du réseau
│   ├── utils.py                 # Utilitaires (chargement graphe, layouts)
│   ├── bfs.py                   # BFS (parcours + arbre)
│   ├── dfs.py                   # DFS (parcours + arbre)
│   ├── prim.py                  # Prim (MST)
│   ├── kruskal.py               # Kruskal (MST)
│   ├── dijkstra.py              # Dijkstra (plus courts chemins)
│   ├── bellman_ford.py          # Bellman-Ford (plus courts chemins)
│   └── floyd_warshall.py        # Floyd-Warshall (plus courts chemins)
├── interface/                   # Interfaces utilisateur (web + desktop)
│   ├── app.py                   # Serveur web Flask (API + routes)
│   ├── gui.py                   # Interface graphique desktop (Tkinter)
│   └── templates/
│       ├── index.html           # Page « Résultat » (visualisations PNG)
│       └── steps.html           # Page « Étape par étape » (graphe + étapes)
└── results/                     # Résultats et visualisations
    ├── BFS/, DFS/, PRIM/, KRUSKAL/
    ├── DIJKSTRA/, BELLMAN_FORD/, FLOYD_WARSHALL/
    ├── metro_network_visualization.png
    └── REFERENCE_GRAPHE_VERIFICATION.txt
```

## Prérequis

- Python 3.7 ou supérieur
- Dépendances (voir `requirements.txt`) :
  - numpy
  - matplotlib
  - networkx
  - flask (pour l’interface web)

## Installation

1. Installer les dépendances :

```bash
pip install -r requirements.txt
```

2. Générer les données du graphe (si pas déjà fait) :

```bash
python algorithms/graph_network.py
```

## Utilisation

### Exécution principale

```bash
python main.py
```

Menu : 1–2 (génération graphe, visualisation), 3–4 (PCC : Bellman-Ford, Dijkstra), 5–6 (MST : Prim, Kruskal), 7–8 (BFS, DFS), 9 (Floyd-Warshall), 10 (GUI), 0 (quitter).

### Interface graphique (bonus)

```bash
python interface/gui.py
```

Ou option **10** dans le menu de `main.py`.  
La GUI permet d’afficher le graphe (zoom, pan), de choisir la station de départ (0–18) et de lancer chaque algorithme : Bellman-Ford, Dijkstra, Floyd-Warshall (matrice toutes paires), Prim, Kruskal, BFS, DFS. Une option **« Poids négatif (Bellman) »** permet d’afficher le graphe cas contrôlé à droite ; lorsque vous sélectionnez Bellman-Ford, ce graphe est utilisé automatiquement.

### Serveur Web

```bash
python interface/app.py
```

- **Démo :** [https://graph-metro-network.onrender.com/](https://graph-metro-network.onrender.com/)
- **Local :** http://localhost:5000  
- **Réseau local :** http://\<IP_DE_LA_MACHINE\>:5000  

Deux onglets accessibles via la barre de navigation sous le titre :

- **Résultat** (`/`) : visualisations en image (graphe métro, BFS, DFS, Prim, Kruskal, Dijkstra, Bellman-Ford, **Floyd-Warshall**). Choisissez la station de départ et un algorithme ; la colonne de droite affiche le réseau métro (Normal ou Poids négatif pour Bellman). Floyd-Warshall affiche la matrice des distances (toutes paires) et la station la plus centrale.
- **Étape par étape** (`/steps`) : pour chaque algorithme (BFS, DFS, Dijkstra, Bellman-Ford, Floyd-Warshall, Prim, Kruskal), affichage pas à pas avec **Précédent** / **Suivant**. Pour les parcours et PCC/MST : graphe avec le même placement que « Résultat ». Pour Floyd-Warshall : affichage de la matrice des distances à chaque étape (intermédiaire k).

API optionnelle : `GET /api/steps/<algo>?start=10` retourne les étapes en JSON (graphe + positions des nœuds + liste d’étapes).

## Développement (optionnel)

Le fichier `pyproject.toml` contient la config **Ruff** (lint/format) et les métadonnées du projet. **Poetry n’est pas requis** : les dépendances restent gérées par `pip install -r requirements.txt`.

Pour lint et format du code :

```bash
pip install ruff
ruff check .
ruff format .
```

- **Docstrings** : modules et classes principaux documentés (niveau simple).
- **`.gitignore`** : ignore `__pycache__/`, venv, IDE, `.ruff_cache/`, `.env`, `*.log`, fichiers OS.

## Notes

- Le graphe principal est **non orienté** ; il existe toujours un chemin entre deux stations.
- Poids des arêtes (graphe principal) = temps de trajet en minutes (1–4 minutes).
- **Bellman-Ford** : le graphe `metro_network_bellman` a une arête (3, 8) de poids -1 ; Bellman-Ford détecte le cycle négatif et affiche « Cycle de poids négatif détecté ».
