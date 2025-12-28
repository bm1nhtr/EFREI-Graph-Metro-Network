import numpy as np

"""
Graphe Réseau Métro - Non orienté
Représente un système de métro avec des stations et des connexions.
Nœuds : Stations de métro (numérotées de 0 à 27)
Arêtes : Connexions entre stations avec poids (temps de trajet en minutes)
Le graphe est NON ORIENTÉ - on peut voyager dans les deux sens
"""

# Réseau Métro : 28 stations (0-27)
# Format : [station_a, station_b, temps_trajet_minutes]
# Comme c'est non orienté, chaque arête représente une connexion bidirectionnelle

metro_network = np.array([
    # Ligne 1 (Ligne principale - 10 stations)
    [0, 1, 3],   # Station 0 -> Station 1 : 3 minutes
    [1, 2, 2],   # Station 1 -> Station 2 : 2 minutes
    [2, 3, 4],   # Station 2 -> Station 3 : 4 minutes
    [3, 4, 3],   # Station 3 -> Station 4 : 3 minutes
    [4, 5, 2],   # Station 4 -> Station 5 : 2 minutes
    [5, 6, 3],   # Station 5 -> Station 6 : 3 minutes
    [6, 7, 4],   # Station 6 -> Station 7 : 4 minutes
    [7, 8, 3],   # Station 7 -> Station 8 : 3 minutes
    [8, 9, 2],   # Station 8 -> Station 9 : 2 minutes
    
    # Ligne 2 (Ligne secondaire - 9 stations)
    [10, 11, 3], # Station 10 -> Station 11 : 3 minutes
    [11, 12, 2], # Station 11 -> Station 12 : 2 minutes
    [12, 13, 4], # Station 12 -> Station 13 : 4 minutes
    [13, 14, 3], # Station 13 -> Station 14 : 3 minutes
    [14, 15, 2], # Station 14 -> Station 15 : 2 minutes
    [15, 16, 3], # Station 15 -> Station 16 : 3 minutes
    [16, 17, 4], # Station 16 -> Station 17 : 4 minutes
    [17, 18, 3], # Station 17 -> Station 18 : 3 minutes
    
    # Ligne 3 (Troisième ligne - 9 stations)
    [19, 20, 3], # Station 19 -> Station 20 : 3 minutes
    [20, 21, 2], # Station 20 -> Station 21 : 2 minutes
    [21, 22, 4], # Station 21 -> Station 22 : 4 minutes
    [22, 23, 3], # Station 22 -> Station 23 : 3 minutes
    [23, 24, 2], # Station 23 -> Station 24 : 2 minutes
    [24, 25, 3], # Station 24 -> Station 25 : 3 minutes
    [25, 26, 4], # Station 25 -> Station 26 : 4 minutes
    [26, 27, 3], # Station 26 -> Station 27 : 3 minutes
    
    # Stations de correspondance (connectent différentes lignes)
    [5, 12, 1],  # Station 5 (Ligne 1) <-> Station 12 (Ligne 2) - Correspondance
    [6, 14, 1],  # Station 6 (Ligne 1) <-> Station 14 (Ligne 2) - Correspondance
    [9, 18, 1],  # Station 9 (Ligne 1) <-> Station 18 (Ligne 2) - Correspondance
    [13, 21, 1], # Station 13 (Ligne 2) <-> Station 21 (Ligne 3) - Correspondance
    [16, 24, 1], # Station 16 (Ligne 2) <-> Station 24 (Ligne 3) - Correspondance
    [7, 20, 1],  # Station 7 (Ligne 1) <-> Station 20 (Ligne 3) - Correspondance
    [8, 23, 1],  # Station 8 (Ligne 1) <-> Station 23 (Ligne 3) - Correspondance
], dtype=int)

# Sauvegarder en format numpy array (.npy)
import os
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(data_dir, exist_ok=True)
np.save(os.path.join(data_dir, 'metro_network.npy'), metro_network)

# Sauvegarder en fichier texte (format lisible)
with open(os.path.join(data_dir, 'metro_network.txt'), 'w', encoding='utf-8') as f:
    f.write("# Graphe Réseau Métro (Non orienté)\n")
    f.write("# Format : station_a station_b temps_trajet_minutes\n")
    f.write("# Total stations : 28 (0-27)\n")
    f.write("# Total connexions : {}\n".format(len(metro_network)))
    f.write("# Le graphe est NON ORIENTÉ - les arêtes représentent des connexions bidirectionnelles\n\n")
    for edge in metro_network:
        f.write("{} {} {}\n".format(edge[0], edge[1], edge[2]))

print("=" * 60)
print("Graphe Réseau Métro Créé")
print("=" * 60)
print(f"Total stations (nœuds) : 28 (0-27)")
print(f"Total connexions (arêtes) : {len(metro_network)}")
print(f"Type de graphe : NON ORIENTÉ")
print(f"\nFichiers créés :")
print(f"  - metro_network.npy (format binaire numpy)")
print(f"  - metro_network.txt (format texte lisible)")
print(f"\nStructure du graphe :")
print(f"  - Ligne 1 : Stations 0-9 (10 stations)")
print(f"  - Ligne 2 : Stations 10-18 (9 stations)")
print(f"  - Ligne 3 : Stations 19-27 (9 stations)")
print(f"  - Stations de correspondance : Connectent différentes lignes")
print(f"  - Poids : Temps de trajet en minutes (1-4 minutes)")
print(f"\nExemples de connexions :")
print(f"  Station 0 <-> Station 1 : {metro_network[0][2]} minutes")
print(f"  Station 5 <-> Station 12 : {metro_network[9][2]} minutes (correspondance)")
print(f"  Station 6 <-> Station 14 : {metro_network[10][2]} minutes (correspondance)")
print("=" * 60)
