import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from algorithms.utils import load_graph_data, LAYOUT_METRO, LAYOUT_METRO_GUI

def visualize_metro_network(fig=None, ax=None):
    """Visualiser le graphe du réseau métro (non orienté).
    Si fig, ax sont fournis (ex: GUI), dessine dessus et ne sauvegarde pas. Sinon crée, sauvegarde et ferme."""
    graph_data = load_graph_data('metro_network.npy')
    interactive = fig is not None and ax is not None
    if not interactive:
        fig, ax = plt.subplots(figsize=(16, 12))

    # Créer un graphe non orienté
    G = nx.Graph()
    for edge in graph_data:
        source, target, weight = edge
        G.add_edge(source, target, weight=weight)
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
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
    
    # Poids sur les arêtes (pour vérification manuelle) — répartis pour limiter les chevauchements
    edge_list = list(G.edges())
    edge_labels = {(u, v): str(G[u][v]["weight"]) for u, v in edge_list}
    for edges_sub, label_pos in [(edge_list[0::3], 0.25), (edge_list[1::3], 0.5), (edge_list[2::3], 0.75)]:
        sub_labels = {e: edge_labels[e] for e in edges_sub if e in edge_labels}
        if sub_labels:
            nx.draw_networkx_edge_labels(G, pos, sub_labels, font_size=10, label_pos=label_pos, ax=ax,
                                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))
    
    # Dessiner les labels pour tous les nœuds
    labels = {node: str(node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)
    
    # Ajouter titre et description
    ax.set_title('Graphe Réseau Métro (Non orienté)\n19 Stations, 20 Connexions', 
                fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Ajouter légende (en GUI : à l'extérieur pour ne pas chevaucher le graphe)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', label='Station Métro'),
        Patch(facecolor='lightcoral', label='Connexion (Arête)'),
    ]
    if interactive:
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10, bbox_to_anchor=(1.02, 1))
    else:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Barre de couleur (échelle des poids = temps de trajet en minutes)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=1, vmax=4))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Poids = Temps de trajet (minutes)', rotation=270, labelpad=20, fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # Description sous le graphe (seulement en export)
    if not interactive:
        fig.text(0.5, 0.02,
                 'Les connexions sont colorées selon leur poids : plus foncé = temps de trajet plus long (1 à 4 min).',
                 ha='center', fontsize=10, style='italic', color='gray')

    if interactive:
        fig.subplots_adjust(**LAYOUT_METRO_GUI)
    else:
        fig.subplots_adjust(**LAYOUT_METRO)
    if not interactive:
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, 'metro_network_visualization.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Visualisation du réseau métro sauvegardée sous '{output_path}'")
        plt.close()
    return fig

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

