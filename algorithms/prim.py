"""
Algorithme de Prim - Arbre couvrant de poids minimum (MST).

Sur le graphe non orienté du réseau métro (poids = temps de trajet).
Implémentation avec liste d'arêtes candidates triée à chaque itération
(sans tas de priorité) : complexité temporelle O(V × E log E).
Avec tas binaire : O(E log V).
"""

import os

import matplotlib.pyplot as plt
import networkx as nx

from algorithms.utils import (
    EXPORT_DPI,
    LAYOUT_ARBRE,
    LAYOUT_ARBRE_GUI,
    LAYOUT_METRO,
    LAYOUT_METRO_GUI,
    SAVEFIG_PNG_OPTIONS,
    standardize_path,
)


class Prim:
    """Arbre couvrant de poids minimum (MST) par l'algorithme de Prim (à partir d'une racine)."""

    def __init__(self, graph_data):
        """Initialise avec les données du graphe (tableau d'arêtes [u, v, poids])."""
        self.graph_data = graph_data

    def get_edges_with_weights(self):
        """Retourne la liste des arêtes avec poids (chaque arête une seule fois pour graphe non orienté).
        Complexité : O(E).
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
        """Retourne les voisins d'un nœud avec le poids de l'arête. Complexité : O(E)."""
        # node (int): Nœud dont on veut les voisins.
        # Returns: list [(voisin, poids), ...]
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

        Complexité : O(V × E log E) (tri de la liste candidates à chaque itération).
        Avec tas de priorité : O(E log V).

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

        # Arêtes candidates (frontière) : (poids, u, v) ; tri à chaque tour → O(E log E) par tour
        candidates = []
        for v, w in self.get_neighbors_with_weights(start_node):
            candidates.append((w, start_node, v))

        while candidates:
            candidates.sort(key=lambda x: x[0])  # O(|candidates| log |candidates|)
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

    def prim_mst_steps(self, start_node: int):
        """Prim étape par étape : yield à chaque arête ajoutée au MST (pour visualisation web). Même complexité que prim_mst."""
        start_node = int(start_node)
        in_mst = {start_node}
        mst_edges = []
        total_weight = 0.0
        candidates = []
        for v, w in self.get_neighbors_with_weights(start_node):
            candidates.append((w, start_node, v))

        step_index = 0
        yield {
            "step_index": step_index,
            "description": f"Départ : nœud {start_node} dans le MST. Arêtes candidates depuis {start_node}.",
            "mst_edges": [],
            "total_weight": 0,
            "in_mst": [start_node],
        }
        step_index += 1

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
            yield {
                "step_index": step_index,
                "description": f"Arête ({u}, {v}) de poids {w} ajoutée. Poids total = {total_weight}.",
                "mst_edges": [[a, b, c] for a, b, c in mst_edges],
                "total_weight": total_weight,
                "in_mst": list(in_mst),
            }
            step_index += 1

        yield {
            "step_index": step_index,
            "description": f"Fin. MST avec {len(mst_edges)} arêtes, poids total = {total_weight}.",
            "mst_edges": [[a, b, c] for a, b, c in mst_edges],
            "total_weight": total_weight,
            "in_mst": list(in_mst),
        }

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

    def visualiser_mst(
        self, mst_edges, total_weight, start_node, file_name="prim_visualization.png", fig=None
    ):
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
        nx.draw_networkx_nodes(G, pos, node_color="lightgray", node_size=400, alpha=0.6, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color="lightgray", width=1.5, alpha=0.4, ax=ax)

        # Arêtes du MST en bleu
        mst_edge_set = set()
        for u, v, _ in mst_edges:
            mst_edge_set.add((u, v))
            mst_edge_set.add((v, u))
        edgelist_mst = [
            (u, v) for u, v in G.edges() if (u, v) in mst_edge_set or (v, u) in mst_edge_set
        ]
        nx.draw_networkx_edges(
            G, pos, edgelist=edgelist_mst, edge_color="blue", width=5, alpha=0.9, ax=ax
        )

        # Poids sur les arêtes (pour vérification manuelle) — répartis pour limiter les chevauchements
        edge_list = list(G.edges())
        edge_labels = {(u, v): str(G[u][v]["weight"]) for u, v in edge_list}
        for edges_sub, label_pos in [
            (edge_list[0::3], 0.25),
            (edge_list[1::3], 0.5),
            (edge_list[2::3], 0.75),
        ]:
            sub_labels = {e: edge_labels[e] for e in edges_sub if e in edge_labels}
            if sub_labels:
                nx.draw_networkx_edge_labels(
                    G,
                    pos,
                    sub_labels,
                    font_size=10,
                    label_pos=label_pos,
                    ax=ax,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85),
                )

        # Nœuds du MST
        mst_nodes = set()
        for u, v, _ in mst_edges:
            mst_nodes.add(u)
            mst_nodes.add(v)
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=list(mst_nodes),
            node_color="steelblue",
            node_size=600,
            alpha=0.9,
            ax=ax,
            edgecolors="black",
            linewidths=2,
        )
        # Racine en rouge pour contraster avec les bleus
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

        labels = {node: str(node) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)

        ax.set_title(
            f"MST (Prim) - Racine: Station {start_node}\nPoids total: {total_weight} min | Arêtes: {len(mst_edges)}",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.axis("off")

        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="red", label=f"Racine ({start_node})"),
            Patch(facecolor="steelblue", label="Stations dans le MST"),
            Patch(facecolor="lightgray", label="Autres stations"),
            Patch(facecolor="blue", edgecolor="blue", label="Arêtes du MST"),
        ]
        if interactive:
            ax.legend(
                handles=legend_elements, loc="upper left", fontsize=10, bbox_to_anchor=(1.02, 1)
            )
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
            plt.savefig(output_path, dpi=EXPORT_DPI, bbox_inches="tight", **SAVEFIG_PNG_OPTIONS)
            print(f"[OK] Visualisation Prim sauvegardée dans: {output_path}")
            plt.close()
        return fig

    def visualiser_arbre_mst(
        self, mst_edges, total_weight, start_node, file_name="prim_tree_visualization.png", fig=None
    ):
        """Visualise uniquement l'arbre MST (Prim) en hiérarchie, racine = start_node. Si fig fourni (GUI), dessine dessus et ne sauvegarde pas."""
        if not mst_edges:
            return fig
        G_tree = nx.Graph()
        for u, v, w in mst_edges:
            G_tree.add_edge(u, v, weight=w)
        # Niveaux par BFS depuis la racine
        levels = {start_node: 0}
        queue = [start_node]
        while queue:
            node = queue.pop(0)
            for neighbor in G_tree.neighbors(node):
                if neighbor not in levels:
                    levels[neighbor] = levels[node] + 1
                    queue.append(neighbor)
        max_level = max(levels.values())
        nodes_by_level = {}
        for n, lev in levels.items():
            nodes_by_level.setdefault(lev, []).append(n)
        pos = {}
        for lev in range(max_level + 1):
            nodes_in_level = nodes_by_level.get(lev, [])
            y_pos = max_level - lev
            for idx, n in enumerate(nodes_in_level):
                x_pos = idx - (len(nodes_in_level) - 1) / 2
                pos[n] = (x_pos, y_pos)

        interactive = fig is not None
        if not interactive:
            fig, ax = plt.subplots(figsize=(14, 10))
        else:
            fig.clear()
            ax = fig.add_subplot(111)

        nx.draw_networkx_edges(
            G_tree, pos, edge_color="blue", width=3, alpha=0.9, ax=ax
        )
        edge_labels = {(u, v): str(G_tree[u][v]["weight"]) for u, v in G_tree.edges()}
        nx.draw_networkx_edge_labels(
            G_tree, pos, edge_labels, font_size=10, ax=ax,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85),
        )
        nx.draw_networkx_nodes(
            G_tree, pos, node_color="steelblue", node_size=600, alpha=0.9, ax=ax,
            edgecolors="black", linewidths=2,
        )
        nx.draw_networkx_nodes(
            G_tree, pos, nodelist=[start_node], node_color="red", node_size=800, alpha=1.0, ax=ax,
            edgecolors="darkred", linewidths=4,
        )
        nx.draw_networkx_labels(G_tree, pos, {n: str(n) for n in G_tree.nodes()}, font_size=10, ax=ax)
        ax.set_title(
            f"Arbre MST (Prim) - Racine: Station {start_node}\nPoids total: {total_weight} min | {len(mst_edges)} arêtes",
            fontsize=14, fontweight="bold", pad=20,
        )
        ax.axis("off")
        from matplotlib.patches import Patch
        ax.legend(
            handles=[
                Patch(facecolor="red", label=f"Racine ({start_node})"),
                Patch(facecolor="steelblue", label="Stations du MST"),
                Patch(facecolor="blue", edgecolor="blue", label="Arêtes du MST"),
            ],
            loc="upper right", fontsize=9,
        )
        if interactive:
            fig.subplots_adjust(**LAYOUT_ARBRE_GUI)
        else:
            fig.subplots_adjust(**LAYOUT_ARBRE)
            plt.tight_layout()
        if not interactive:
            results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "PRIM")
            os.makedirs(results_dir, exist_ok=True)
            output_path = standardize_path(os.path.join(results_dir, file_name))
            plt.savefig(output_path, dpi=EXPORT_DPI, bbox_inches="tight", **SAVEFIG_PNG_OPTIONS)
            print(f"[OK] Arbre MST Prim sauvegardé dans: {output_path}")
            plt.close()
        return fig
