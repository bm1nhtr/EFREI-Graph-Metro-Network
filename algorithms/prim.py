"""
Algorithme de Prim - Arbre couvrant de poids minimum (MST)
Sur le graphe non orienté du réseau métro (poids = temps de trajet).
"""

from algorithms.utils import standardize_path, LAYOUT_METRO, LAYOUT_METRO_GUI
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class Prim:
    def __init__(self, graph_data):
        self.graph_data = graph_data

    def get_edges_with_weights(self):
        """Retourne la liste des arêtes avec poids (chaque arête une seule fois pour graphe non orienté).
        Returns:
            list: [(u, v, weight), ...]
        """
        edges = []
        seen = set()
        for edge in self.graph_data:
            u, v, w = int(edge[0]), int(edge[1]), float(edge[2])
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                edges.append((u, v, w))
        return edges

    def get_neighbors_with_weights(self, node: int):
        """Retourne les voisins d'un nœud avec le poids de l'arête.
        Args:
            node (int): Nœud dont on veut les voisins.
        Returns:
            list: [(voisin, poids), ...]
        """
        node = int(node)
        result = []
        for edge in self.graph_data:
            source, target, weight = int(edge[0]), int(edge[1]), float(edge[2])
            if source == node:
                result.append((target, weight))
            elif target == node:
                result.append((source, weight))
        return result

    def prim_mst(self, start_node: int):
        """
        Calcule un arbre couvrant de poids minimum (MST) par l'algorithme de Prim.
        Args:
            start_node (int): Nœud de départ (racine de l'arbre).
        Returns:
            tuple: (mst_edges, total_weight)
                - mst_edges: liste de (u, v, weight)
                - total_weight: poids total du MST
        """
        start_node = int(start_node)
        in_mst = {start_node}
        mst_edges = []
        total_weight = 0.0

        # Arêtes candidates: (poids, u, v) pour tri
        candidates = []
        for v, w in self.get_neighbors_with_weights(start_node):
            candidates.append((w, start_node, v))

        while candidates:
            candidates.sort(key=lambda x: x[0])
            w, u, v = candidates.pop(0)
            if v in in_mst:
                continue
            in_mst.add(v)
            mst_edges.append((u, v, w))
            total_weight += w
            for neighbor, nw in self.get_neighbors_with_weights(v):
                if neighbor not in in_mst:
                    candidates.append((nw, v, neighbor))

        return mst_edges, total_weight

    def sauvegarder_resultats(self, mst_edges, total_weight, file_name="prim_result.txt"):
        """Sauvegarde les résultats du MST (arêtes et poids total).
        Args:
            mst_edges (list): Liste des arêtes du MST (u, v, weight).
            total_weight (float): Poids total du MST.
            file_name (str): Nom du fichier de sortie.
        """
        results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "PRIM")
        os.makedirs(results_dir, exist_ok=True)
        output_path = standardize_path(os.path.join(results_dir, file_name))

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Résultats Algorithme de Prim - Arbre couvrant minimum (MST)\n")
            f.write(f"# Poids total du MST: {total_weight}\n")
            f.write(f"# Nombre d'arêtes: {len(mst_edges)}\n\n")
            f.write("# Arêtes (station_u station_v poids)\n")
            for u, v, w in mst_edges:
                f.write(f"{u} {v} {w}\n")

        print(f"[OK] Résultats Prim sauvegardés dans: {output_path}")

    def visualiser_mst(self, mst_edges, total_weight, start_node, file_name="prim_visualization.png", fig=None):
        """Visualise le MST sur le graphe du réseau métro. Si fig fourni (GUI), dessine dessus et ne sauvegarde pas."""
        interactive = fig is not None
        if not interactive:
            fig, ax = plt.subplots(figsize=(16, 12))
        else:
            fig.clear()
            ax = fig.add_subplot(111)

        G = nx.Graph()
        for edge in self.graph_data:
            source, target, weight = edge
            G.add_edge(source, target, weight=weight)
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Arêtes du graphe complet (grises)
        all_edges = list(G.edges())
        nx.draw_networkx_nodes(G, pos, node_color="lightgray", node_size=400, alpha=0.6, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color="lightgray", width=1.5, alpha=0.4, ax=ax)

        # Arêtes du MST en bleu
        mst_edge_set = set()
        for u, v, _ in mst_edges:
            mst_edge_set.add((u, v))
            mst_edge_set.add((v, u))
        edgelist_mst = [(u, v) for u, v in G.edges() if (u, v) in mst_edge_set or (v, u) in mst_edge_set]
        nx.draw_networkx_edges(G, pos, edgelist=edgelist_mst, edge_color="blue", width=5, alpha=0.9, ax=ax)

        # Poids sur les arêtes (pour vérification manuelle) — répartis pour limiter les chevauchements
        edge_list = list(G.edges())
        edge_labels = {(u, v): str(G[u][v]["weight"]) for u, v in edge_list}
        for edges_sub, label_pos in [(edge_list[0::3], 0.25), (edge_list[1::3], 0.5), (edge_list[2::3], 0.75)]:
            sub_labels = {e: edge_labels[e] for e in edges_sub if e in edge_labels}
            if sub_labels:
                nx.draw_networkx_edge_labels(G, pos, sub_labels, font_size=10, label_pos=label_pos, ax=ax,
                                             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))

        # Nœuds du MST
        mst_nodes = set()
        for u, v, _ in mst_edges:
            mst_nodes.add(u)
            mst_nodes.add(v)
        nx.draw_networkx_nodes(G, pos, nodelist=list(mst_nodes), node_color="steelblue", node_size=600, alpha=0.9, ax=ax, edgecolors="black", linewidths=2)
        # Racine en rouge pour contraster avec les bleus
        nx.draw_networkx_nodes(G, pos, nodelist=[start_node], node_color="red", node_size=800, alpha=1.0, ax=ax, edgecolors="darkred", linewidths=4)

        labels = {node: str(node) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)

        ax.set_title(f"MST (Prim) - Racine: Station {start_node}\nPoids total: {total_weight} min | Arêtes: {len(mst_edges)}", fontsize=16, fontweight="bold", pad=20)
        ax.axis("off")

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="red", label=f"Racine ({start_node})"),
            Patch(facecolor="steelblue", label="Stations dans le MST"),
            Patch(facecolor="lightgray", label="Autres stations"),
            Patch(facecolor="blue", edgecolor="blue", label="Arêtes du MST"),
        ]
        if interactive:
            ax.legend(handles=legend_elements, loc="upper left", fontsize=10, bbox_to_anchor=(1.02, 1))
        else:
            ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

        if interactive:
            fig.subplots_adjust(**LAYOUT_METRO_GUI)
        else:
            fig.subplots_adjust(**LAYOUT_METRO)
            plt.tight_layout()
        if not interactive:
            results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "PRIM")
            os.makedirs(results_dir, exist_ok=True)
            output_path = standardize_path(os.path.join(results_dir, file_name))
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"[OK] Visualisation Prim sauvegardée dans: {output_path}")
            plt.close()
        return fig
