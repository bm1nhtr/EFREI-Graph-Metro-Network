import numpy as np
import os

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
    """Charger les données du graphe depuis un fichier numpy
    
    Args:
        file_name (str): Nom du fichier numpy contenant les données du graphe.
    
    Returns:
        np.ndarray: Données du graphe chargées.
    """
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    graph_data = np.load(standardize_path(os.path.join(data_dir, file_name)))
    return graph_data