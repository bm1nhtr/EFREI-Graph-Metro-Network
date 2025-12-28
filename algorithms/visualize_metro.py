import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

def visualize_metro_network():
    """Visualiser le graphe du réseau métro (non orienté)"""
    print("Chargement du réseau métro...")
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    graph_data = np.load(os.path.join(data_dir, 'metro_network.npy'))
    
    # Créer un graphe non orienté
    G = nx.Graph()
    
    # Ajouter les arêtes avec poids
    for edge in graph_data:
        source, target, weight = edge
        G.add_edge(source, target, weight=weight)
    
    # Créer la disposition
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Dessiner les nœuds
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=500, alpha=0.9, ax=ax)
    
    # Dessiner les arêtes avec couleurs basées sur le poids
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    # Carte de couleurs pour les arêtes (plus foncé = temps de trajet plus long)
    edge_colors = plt.cm.Reds([w/4.0 for w in weights])
    
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, 
                          width=3, alpha=0.7, ax=ax)
    
    # Dessiner les labels pour tous les nœuds
    labels = {node: str(node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)
    
    # Ajouter titre et informations
    ax.set_title('Graphe Réseau Métro (Non orienté)\n28 Stations, 32 Connexions', 
                fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Ajouter légende
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', label='Station Métro'),
        Patch(facecolor='lightcoral', label='Connexion (Arête)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, 'metro_network_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Visualisation du réseau métro sauvegardée sous '{output_path}'")
    plt.close()

if __name__ == "__main__":
    print("=" * 70)
    print("Outil de Visualisation du Réseau Métro")
    print("=" * 70)
    print()
    
    try:
        visualize_metro_network()
        print()
        print("=" * 70)
        print("Visualisation créée avec succès !")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"Erreur : {e}")
        print("Assurez-vous que metro_network.npy existe. Exécutez graph_network.py d'abord.")
    except ImportError as e:
        print(f"Erreur : Bibliothèque manquante - {e}")
        print("Veuillez installer les packages requis : pip install matplotlib networkx numpy")
    except Exception as e:
        print(f"Erreur : {e}")

