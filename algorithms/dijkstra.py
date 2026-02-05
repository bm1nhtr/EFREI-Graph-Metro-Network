"""
Algorithme de Dijkstra - Plus courts chemins depuis une source
Fonctionne sur graphe pondéré à poids positifs.
Sur le réseau métro : graphe non orienté.
"""

from algorithms.utils import standardize_path, LAYOUT_METRO, LAYOUT_METRO_GUI
import os
import heapq
import matplotlib.pyplot as plt
import networkx as nx


class Dijkstra:
    def __init__(self, graph_data):
        # Stocke les données du graphe (arêtes + poids)
        self.graph_data = graph_data

    def get_neighbors_with_weights(self, node: int):
        """Retourne les voisins d’un nœud avec les poids associés."""
        neighbors = []
        node = int(node)

        # Parcourir les arêtes pour trouver les voisins pondérés
        for edge in self.graph_data:
            u, v, w = int(edge[0]), int(edge[1]), float(edge[2])
            if u == node:
                neighbors.append((v, w))
            elif v == node:
                neighbors.append((u, w))

        return neighbors

    def get_nodes(self):
        """Retourne l’ensemble des nœuds du graphe."""
        nodes = set()
        for edge in self.graph_data:
            nodes.add(int(edge[0]))
            nodes.add(int(edge[1]))
        return nodes

    def dijkstra(self, start_node: int):
        """
        Calcule les plus courts chemins depuis start_node avec Dijkstra.
        Returns:
            tuple: (distances, predecessors)
        """
        start_node = int(start_node)
        nodes = self.get_nodes()

        # Initialiser distances et prédécesseurs
        distances = {n: float("inf") for n in nodes}
        predecessors = {n: None for n in nodes}
        distances[start_node] = 0

        # File de priorité (distance, nœud)
        priority_queue = [(0, start_node)]

        # Boucle principale de Dijkstra
        while priority_queue:
            current_dist, current_node = heapq.heappop(priority_queue)

            # Ignorer les entrées obsolètes
            if current_dist > distances[current_node]:
                continue

            # Parcourir les voisins du nœud courant
            for neighbor, weight in self.get_neighbors_with_weights(current_node):
                new_dist = current_dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    predecessors[neighbor] = current_node
                    heapq.heappush(priority_queue, (new_dist, neighbor))

        return distances, predecessors

    def sauvegarder_resultats(self, distances, predecessors, start_node, file_name="dijkstra_result.txt"):
        """Sauvegarde les distances et les chemins calculés."""
        results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "DIJKSTRA")
        os.makedirs(results_dir, exist_ok=True)
        output_path = standardize_path(os.path.join(results_dir, file_name))

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Résultats Dijkstra - Plus courts chemins depuis la source\n")
            f.write(f"# Source: Station {start_node}\n\n")
            f.write("# Noeud\tDistance\tPredecesseur\n")

            for node in sorted(distances.keys()):
                d = distances[node]
                d_str = str(int(d)) if d != float("inf") else "inf"
                pred = predecessors[node]
                pred_str = str(pred) if pred is not None else "-"
                f.write(f"{node}\t{d_str}\t{pred_str}\n")

        print(f"[OK] Résultats Dijkstra sauvegardés dans: {output_path}")

    def _shortest_path_tree_edges(self, predecessors):
        """Construit les arêtes de l’arbre des plus courts chemins."""
        edges = set()
        for node, pred in predecessors.items():
            if pred is not None:
                edges.add((pred, node))
                edges.add((node, pred))  # non orienté pour affichage
        return edges

    def visualiser_parcours(self, distances, predecessors, start_node, file_name="dijkstra_visualization.png", fig=None):
        """Visualise l’arbre des plus courts chemins de Dijkstra."""
        interactive = fig is not None
        if not interactive:
            fig, ax = plt.subplots(figsize=(16, 12))
        else:
            fig.clear()
            ax = fig.add_subplot(111)

        # Construire le graphe NetworkX
        G = nx.Graph()
        for edge in self.graph_data:
            u, v, w = edge
            G.add_edge(u, v, weight=w)

        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Dessiner le graphe de base
        nx.draw_networkx_nodes(G, pos, node_color="lightgray", node_size=400, alpha=0.6, ax=ax)
        edges = G.edges()
        weights = [G[u][v]["weight"] for u, v in edges]
        edge_colors = plt.cm.Reds([w / 4.0 for w in weights])
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2.5, alpha=0.5, ax=ax)

        # Arêtes de l’arbre des plus courts chemins
        tree_edges = self._shortest_path_tree_edges(predecessors)
        edgelist_tree = [(u, v) for u, v in G.edges() if (u, v) in tree_edges]
        if edgelist_tree:
            nx.draw_networkx_edges(G, pos, edgelist=edgelist_tree,
                                   edge_color="blue", width=5, alpha=0.9, ax=ax)

        # Coloration des nœuds selon la distance
        max_d = max((d for d in distances.values() if d != float("inf")), default=1)
        node_colors = []
        for n in G.nodes():
            if distances[n] == float("inf"):
                node_colors.append((0.9, 0.9, 0.9, 0.8))
            else:
                node_colors.append(plt.cm.viridis(distances[n] / max_d))

        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                               node_size=600, alpha=0.9, ax=ax,
                               edgecolors="black", linewidths=2)

        # Mettre en évidence la source
        nx.draw_networkx_nodes(G, pos, nodelist=[start_node],
                               node_color="red", node_size=800,
                               alpha=1.0, ax=ax,
                               edgecolors="darkred", linewidths=4)

        # Labels
        labels = {node: str(node) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)

        # Titre
        title = f"Dijkstra - Source: Station {start_node}\nPlus courts chemins"
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.axis("off")

        # Légende
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="red", label=f"Source ({start_node})"),
            Patch(facecolor="steelblue", label="Nœuds (couleur = distance)"),
            Patch(facecolor="blue", edgecolor="blue", label="Arêtes des PCC"),
        ]

        if interactive:
            ax.legend(handles=legend_elements, loc="upper left",
                      fontsize=10, bbox_to_anchor=(1.02, 1))
        else:
            ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

        # Barre de couleur
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                   norm=plt.Normalize(vmin=0, vmax=max_d))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="vertical",
                            fraction=0.02, pad=0.04)
        cbar.set_label("Distance depuis la source", rotation=270, labelpad=20)

        if interactive:
            fig.subplots_adjust(**LAYOUT_METRO_GUI)
        else:
            fig.subplots_adjust(**LAYOUT_METRO)
            plt.tight_layout()

        if not interactive:
            results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "DIJKSTRA")
            os.makedirs(results_dir, exist_ok=True)
            output_path = standardize_path(os.path.join(results_dir, file_name))
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"[OK] Visualisation Dijkstra sauvegardée dans: {output_path}")
            plt.close()

        return fig