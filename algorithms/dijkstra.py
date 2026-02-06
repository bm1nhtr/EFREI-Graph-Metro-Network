"""
Algorithme de Dijkstra - Plus courts chemins depuis une source
Fonctionne sur graphe pondéré à poids positifs.
Sur le réseau métro : graphe non orienté.
"""

import heapq
import os

import matplotlib.pyplot as plt
import networkx as nx

from algorithms.utils import LAYOUT_METRO, LAYOUT_METRO_GUI, standardize_path


class Dijkstra:
    """Plus courts chemins depuis une source (graphe à poids positifs)."""

    def __init__(self, graph_data):
        """Initialise avec les données du graphe (tableau d'arêtes [u, v, poids])."""
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
        Calcule les plus courts chemins depuis start_node (Dijkstra).

        Returns:
            tuple: (distances dict, predecessors dict).
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

    def dijkstra_steps(self, start_node: int):
        """Dijkstra étape par étape : yield à chaque extraction de nœud (pour visualisation web)."""
        start_node = int(start_node)
        nodes = self.get_nodes()
        distances = {n: float("inf") for n in nodes}
        predecessors = {n: None for n in nodes}
        distances[start_node] = 0
        priority_queue = [(0, start_node)]
        step_index = 0

        def dist_repr(d):
            return int(d) if d != float("inf") else "inf"

        yield {
            "step_index": step_index,
            "description": f"Initialisation : source = {start_node}, distances[{start_node}] = 0.",
            "distances": {k: dist_repr(v) for k, v in distances.items()},
            "predecessors": {k: v for k, v in predecessors.items()},
            "current_node": None,
        }
        step_index += 1

        while priority_queue:
            current_dist, current_node = heapq.heappop(priority_queue)
            if current_dist > distances[current_node]:
                continue
            for neighbor, weight in self.get_neighbors_with_weights(current_node):
                new_dist = current_dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    predecessors[neighbor] = current_node
                    heapq.heappush(priority_queue, (new_dist, neighbor))
            yield {
                "step_index": step_index,
                "description": f"Nœud {current_node} extrait (dist = {dist_repr(distances[current_node])}). Relaxation des voisins.",
                "distances": {k: dist_repr(v) for k, v in distances.items()},
                "predecessors": {k: v for k, v in predecessors.items()},
                "current_node": current_node,
            }
            step_index += 1

        yield {
            "step_index": step_index,
            "description": "Fin. Plus courts chemins calculés depuis la source.",
            "distances": {k: dist_repr(v) for k, v in distances.items()},
            "predecessors": {k: v for k, v in predecessors.items()},
            "current_node": None,
        }

    def _get_shortest_path(self, predecessors, start_node: int, end_node: int):
        """Reconstruit l'unique plus court chemin (source -> noeud) pour Dijkstra."""
        if end_node == start_node:
            return [start_node]
        path = []
        current = end_node
        while current is not None:
            path.append(current)
            current = predecessors.get(current)
        path.reverse()
        return path if path and path[0] == start_node else []

    def sauvegarder_resultats(
        self, distances, predecessors, start_node, file_name="dijkstra_result.txt"
    ):
        """Sauvegarde distances, predecesseurs et chemins (source -> noeud) dans results/DIJKSTRA/."""
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

            f.write("\n# Chemins (source -> noeud)\n")
            for node in sorted(distances.keys()):
                if node == start_node:
                    f.write(f"{start_node} -> {start_node}: [{start_node}]\n")
                    continue
                path = self._get_shortest_path(predecessors, start_node, node)
                if not path:
                    continue
                dist_val = int(distances[node])
                f.write(
                    f"{start_node} -> {node} (dist={dist_val})  {' -> '.join(map(str, path))}\n"
                )

        print(f"[OK] Résultats Dijkstra sauvegardés dans: {output_path}")

    def _shortest_path_tree_edges(self, predecessors):
        """Construit les arêtes de l’arbre des plus courts chemins."""
        edges = set()
        for node, pred in predecessors.items():
            if pred is not None:
                edges.add((pred, node))
                edges.add((node, pred))  # non orienté pour affichage
        return edges

    def visualiser_parcours(
        self, distances, predecessors, start_node, file_name="dijkstra_visualization.png", fig=None
    ):
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
            nx.draw_networkx_edges(
                G, pos, edgelist=edgelist_tree, edge_color="blue", width=5, alpha=0.9, ax=ax
            )

        # Coloration des nœuds selon la distance
        max_d = max((d for d in distances.values() if d != float("inf")), default=1)
        node_colors = []
        for n in G.nodes():
            if distances[n] == float("inf"):
                node_colors.append((0.9, 0.9, 0.9, 0.8))
            else:
                node_colors.append(plt.cm.viridis(distances[n] / max_d))

        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_colors,
            node_size=600,
            alpha=0.9,
            ax=ax,
            edgecolors="black",
            linewidths=2,
        )

        # Mettre en évidence la source
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[start_node],
            node_color="red",
            node_size=800,
            alpha=1.0,
            ax=ax,
            edgecolors="darkred",
            linewidths=4,
        )

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
            ax.legend(
                handles=legend_elements, loc="upper left", fontsize=10, bbox_to_anchor=(1.02, 1)
            )
        else:
            ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

        # Barre de couleur
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=max_d))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)
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
