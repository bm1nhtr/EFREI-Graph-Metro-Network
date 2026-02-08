"""
Utilitaires partagés par les algorithmes et visualisations.

Constantes de layout (Matplotlib) et fonctions de chargement/chemins.
"""
import os

import numpy as np

# --- Layout Matplotlib (marges, position légende) ---
# Réseau métro : export image
LAYOUT_METRO = dict(left=0.062, bottom=0, right=0.948, top=0.886, wspace=0.217, hspace=0.217)
# Réseau métro en GUI : marge droite pour légende
LAYOUT_METRO_GUI = dict(left=0.062, bottom=0, right=0.70, top=0.886, wspace=0.217, hspace=0.217)
# Arbres (BFS, DFS, MST) : format vertical
LAYOUT_ARBRE = dict(left=0.036, bottom=0.05, right=0.676, top=0.829, wspace=0.217, hspace=0.217)
LAYOUT_ARBRE_GUI = dict(left=0.036, bottom=0.05, right=0.676, top=0.829, wspace=0.217, hspace=0.217)
# Floyd-Warshall : matrice en GUI
LAYOUT_FLOYD_GUI = dict(left=0.062, bottom=0.107, right=0.7, top=0.886, wspace=0.217, hspace=0.217)

# Export images : DPI réduit pour alléger les PNG (rapport LaTeX)
EXPORT_DPI = 100
SAVEFIG_PNG_OPTIONS = {"pil_kwargs": {"compress_level": 6}}


def standardize_path(path):
    """Standardiser le chemin pour éviter les incohérences entre systèmes d'exploitation (ex. Windows, MacOS).
    Le chemin peut contenir des caractères comme '/' et '\'.

    Args:
        path (str): Chemin à standardiser.

    Returns:
        str: Chemin standardisé.
    """
    return os.path.normpath(path)


def load_graph_data(file_name):
    """Charger les données du graphe depuis data/<file_name> (format numpy, tableau N×3 : u, v, poids).

    Args:
        file_name (str): Nom du fichier (ex. metro_network.npy).

    Returns:
        np.ndarray: Données du graphe (arêtes avec poids).
    """
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    graph_data = np.load(standardize_path(os.path.join(data_dir, file_name)))
    return graph_data
