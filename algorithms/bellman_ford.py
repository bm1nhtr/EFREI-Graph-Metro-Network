"""
Algorithme de Bellman-Ford - Plus courts chemins depuis une source
Fonctionne sur graphe avec poids (éventuellement négatifs).
Sur le réseau métro : chaque arête non orientée est traitée comme deux arcs (u->v, v->u).
"""

import os

import matplotlib.pyplot as plt
import networkx as nx

from algorithms.utils import LAYOUT_METRO, LAYOUT_METRO_GUI, standardize_path


class BellmanFord:
    def __init__(self, graph_data):
        self.graph_data = graph_data

    def get_directed_edges(self):
        """Pour Bellman-Ford on traite le graphe comme orienté : chaque arête (u,v,w) donne deux arcs (u,v,w) et (v,u,w).
        Returns:
            list: [(source, target, weight), ...]
        """
        edges = []
        for edge in self.graph_data:
            u, v, w = int(edge[0]), int(edge[1]), float(edge[2])
            edges.append((u, v, w))
            edges.append((v, u, w))
        return edges

    def get_nodes(self):
        """Retourne l'ensemble des nœuds du graphe."""
        nodes = set()
        for edge in self.graph_data:
            nodes.add(int(edge[0]))
            nodes.add(int(edge[1]))
        return nodes

    def bellman_ford(self, start_node: int):
        """
        Calcule les plus courts chemins depuis start_node vers tous les nœuds (Bellman-Ford).
        Stocke tous les prédécesseurs possibles par nœud (tous les PCC).
        Returns:
            tuple: (distances, predecessors, has_negative_cycle)
                - distances: dict {node: distance}
                - predecessors: dict {node: [pred1, pred2, ...]} liste de tous les prédécesseurs
                - has_negative_cycle: bool
        """
        start_node = int(start_node)
        nodes = self.get_nodes()
        edges = self.get_directed_edges()

        distances = {n: float("inf") for n in nodes}
        predecessors = {n: [] for n in nodes}
        distances[start_node] = 0

        # Relaxation (|V|-1) fois
        for _ in range(len(nodes) - 1):
            for u, v, w in edges:
                if distances[u] == float("inf"):
                    continue
                d_new = distances[u] + w
                if d_new < distances[v]:
                    distances[v] = d_new
                    predecessors[v] = [u]
                elif d_new == distances[v] and u not in predecessors[v]:
                    predecessors[v].append(u)

        # Détection cycle de poids négatif
        has_negative_cycle = False
        for u, v, w in edges:
            if distances[u] != float("inf") and distances[u] + w < distances[v]:
                has_negative_cycle = True
                break

        return distances, predecessors, has_negative_cycle

    def bellman_ford_steps(self, start_node: int):
        """Bellman-Ford étape par étape : yield après chaque phase de relaxation (pour visualisation web)."""
        start_node = int(start_node)
        nodes = self.get_nodes()
        edges = self.get_directed_edges()
        distances = {n: float("inf") for n in nodes}
        predecessors = {n: [] for n in nodes}
        distances[start_node] = 0

        def dist_repr(d):
            return int(d) if d != float("inf") else "inf"

        step_index = 0
        yield {
            "step_index": step_index,
            "description": f"Initialisation : source = {start_node}, distances[{start_node}] = 0.",
            "distances": {k: dist_repr(v) for k, v in distances.items()},
            "predecessors": {k: list(v) for k, v in predecessors.items()},
            "phase": 0,
        }
        step_index += 1

        for phase in range(1, len(nodes)):
            updated = False
            for u, v, w in edges:
                if distances[u] == float("inf"):
                    continue
                d_new = distances[u] + w
                if d_new < distances[v]:
                    distances[v] = d_new
                    predecessors[v] = [u]
                    updated = True
                elif d_new == distances[v] and u not in predecessors[v]:
                    predecessors[v].append(u)
                    updated = True
            yield {
                "step_index": step_index,
                "description": f"Phase {phase} : relaxation de toutes les arêtes. Distances mises à jour = {updated}.",
                "distances": {k: dist_repr(v) for k, v in distances.items()},
                "predecessors": {k: list(v) for k, v in predecessors.items()},
                "phase": phase,
            }
            step_index += 1

        has_negative_cycle = False
        for u, v, w in edges:
            if distances[u] != float("inf") and distances[u] + w < distances[v]:
                has_negative_cycle = True
                break
        yield {
            "step_index": step_index,
            "description": "Détection cycle négatif : " + ("oui, cycle détecté." if has_negative_cycle else "non."),
            "distances": {k: dist_repr(v) for k, v in distances.items()},
            "predecessors": {k: list(v) for k, v in predecessors.items()},
            "phase": len(nodes) - 1,
            "has_negative_cycle": has_negative_cycle,
        }

    def get_shortest_path(
        self, predecessors, start_node: int, end_node: int, pred_index: int = 0, max_steps: int = 50
    ):
        """Reconstruit un plus court chemin de start_node à end_node (utilise le pred_index-ième prédécesseur).
        Limite à max_steps pour éviter boucle infinie en cas de cycle négatif.
        Returns:
            list: Liste des nœuds du chemin, ou [] si pas de chemin.
        """
        pred_list = predecessors.get(end_node)
        if not pred_list and end_node != start_node:
            return []
        path = []
        current = end_node
        visited = set()
        steps = 0
        while current is not None and steps < max_steps:
            if current in visited:
                break  # cycle détecté
            visited.add(current)
            path.append(current)
            pred_list = predecessors.get(current)
            if not pred_list:
                break
            idx = min(pred_index, len(pred_list) - 1)
            current = pred_list[idx]
            steps += 1
        path.reverse()
        return path if path and path[0] == start_node else []

    def sauvegarder_resultats(
        self,
        distances,
        predecessors,
        start_node,
        has_negative_cycle,
        file_name="bellman_ford_result.txt",
    ):
        """Sauvegarde les distances et prédécesseurs.
        Args:
            distances (dict): Nœud -> distance depuis la source.
            predecessors (dict): Nœud -> prédécesseur.
            start_node (int): Nœud source.
            has_negative_cycle (bool): Présence d'un cycle négatif.
            file_name (str): Nom du fichier.
        """
        results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "BELLMAN_FORD")
        os.makedirs(results_dir, exist_ok=True)
        output_path = standardize_path(os.path.join(results_dir, file_name))

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Résultats Bellman-Ford - Plus courts chemins depuis la source\n")
            f.write(f"# Source: Station {start_node}\n")
            f.write(f"# Cycle de poids négatif détecté: {has_negative_cycle}\n\n")
            if has_negative_cycle:
                f.write(
                    "# Explication (cas contrôlé avec 1 poids négatif, ex. arête 3-8 = -1) :\n"
                    '# - Un cycle négatif (ex. 3→8→3) permet de diminuer la "distance" à l\'infini.\n'
                    "# - Les PCC n'existent pas : on ne reconstruit donc pas les chemins.\n"
                    '# - Les valeurs "Distance" ci-dessous sont des artefacts des relaxations (elles\n'
                    "#   deviennent très négatives car l'algo utilise le cycle à chaque itération).\n"
                    "#   En théorie, les vrais plus courts chemins seraient -infini pour les nœuds\n"
                    "#   atteignables depuis le cycle.\n\n"
                )
            f.write("# Noeud\tDistance\tPredecesseur(s)\n")
            for node in sorted(distances.keys()):
                d = distances[node]
                pred_list = predecessors.get(node) or []
                d_str = str(int(d)) if d != float("inf") else "inf"
                p_str = ",".join(map(str, pred_list)) if pred_list else "-"
                f.write(f"{node}\t{d_str}\t{p_str}\n")
            if has_negative_cycle:
                f.write("\n# Chemins non reconstruits (cycle de poids négatif détecté).\n")
            else:
                f.write("\n# Chemins (source -> noeud) — tous les PCC quand plusieurs existent\n")
                for node in sorted(distances.keys()):
                    if node == start_node:
                        f.write(f"{start_node} -> {start_node}: [{start_node}]\n")
                        continue
                    pred_list = predecessors.get(node) or []
                    if not pred_list:
                        continue
                    dist_val = int(distances[node])
                    for idx in range(len(pred_list)):
                        path = self.get_shortest_path(
                            predecessors, start_node, node, pred_index=idx
                        )
                        if path:
                            num = f"  Path {idx + 1}: " if len(pred_list) > 1 else "  "
                            f.write(
                                f"{start_node} -> {node} (dist={dist_val}){num}{' -> '.join(map(str, path))}\n"
                            )
                    if len(pred_list) > 1:
                        f.write("\n")

        print(f"[OK] Résultats Bellman-Ford sauvegardés dans: {output_path}")

    def _shortest_path_tree_edges(self, predecessors, start_node):
        """Arêtes du graphe des plus courts chemins (tous les pred -> nœud)."""
        edges = set()
        for node, pred_list in predecessors.items():
            for pred in pred_list or []:
                edges.add((pred, node))
                edges.add((node, pred))  # non orienté pour dessin
        return edges

    def visualiser_parcours(
        self,
        distances,
        predecessors,
        start_node,
        has_negative_cycle,
        file_name="bellman_ford_visualization.png",
        fig=None,
    ):
        """Visualise l'arbre des plus courts chemins. Si fig fourni (GUI), dessine dessus et ne sauvegarde pas."""
        interactive = fig is not None
        if not interactive:
            fig, ax = plt.subplots(figsize=(16, 12))
        else:
            fig.clear()
            ax = fig.add_subplot(111)

        G = nx.Graph()
        for edge in self.graph_data:
            source, target, weight = float(edge[0]), float(edge[1]), float(edge[2])
            G.add_edge(int(source), int(target), weight=weight)
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        nx.draw_networkx_nodes(G, pos, node_color="lightgray", node_size=400, alpha=0.6, ax=ax)
        edges = G.edges()
        weights = [G[u][v]["weight"] for u, v in edges]
        has_neg_w = any(w < 0 for w in weights)
        if has_neg_w:
            w_min, w_max = min(weights), max(weights)
            norm = plt.Normalize(vmin=w_min, vmax=max(w_max, 1e-6))
            edge_colors = plt.cm.RdBu_r(norm(weights))
        else:
            edge_colors = plt.cm.Reds([w / 4.0 for w in weights])
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2.5, alpha=0.5, ax=ax)

        # Arêtes du plus court chemin (arbre)
        tree_edges = self._shortest_path_tree_edges(predecessors, start_node)
        edgelist_tree = [
            (u, v) for u, v in G.edges() if (u, v) in tree_edges or (v, u) in tree_edges
        ]
        if edgelist_tree:
            nx.draw_networkx_edges(
                G, pos, edgelist=edgelist_tree, edge_color="blue", width=5, alpha=0.9, ax=ax
            )

        # Poids sur les arêtes (afficher la valeur réelle, ex. -1 pour cas contrôlé)
        edge_list = list(G.edges())
        edge_labels = {}
        for u, v in edge_list:
            w = G[u][v]["weight"]
            edge_labels[(u, v)] = str(int(w)) if w == int(w) else str(w)
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

        # Nœuds avec couleur selon la distance (gérer distances négatives si cycle)
        nodes_in_tree = set()
        for u, v in tree_edges:
            nodes_in_tree.add(u)
            nodes_in_tree.add(v)
        finite_d = [d for d in distances.values() if d != float("inf")]
        min_d = min(finite_d, default=0)
        max_d = max(finite_d, default=1)
        node_colors = []
        node_list = list(G.nodes())
        for n in node_list:
            if n not in distances or distances[n] == float("inf"):
                node_colors.append((0.9, 0.9, 0.9, 0.8))
            else:
                d = distances[n]
                span = max_d - min_d
                r = (d - min_d) / span if span > 0 else 0.5
                r = max(0.0, min(1.0, r))
                node_colors.append(plt.cm.viridis(r))
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=node_list,
            node_color=node_colors,
            node_size=600,
            alpha=0.9,
            ax=ax,
            edgecolors="black",
            linewidths=2,
        )
        # Source en rouge pour contraster avec les bleus/verts
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

        title = f"Bellman-Ford - Source: Station {start_node}\n"
        title += "Plus courts chemins vers tous les nœuds"
        if has_negative_cycle:
            title += " (cycle négatif détecté!)"
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.axis("off")

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

        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.viridis,
            norm=plt.Normalize(vmin=min_d, vmax=max(max_d, min_d + 1e-6)),
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)
        cbar.set_label("Distance depuis la source", rotation=270, labelpad=20)

        if interactive:
            fig.subplots_adjust(**LAYOUT_METRO_GUI)
        else:
            fig.subplots_adjust(**LAYOUT_METRO)
            plt.tight_layout()
        if not interactive:
            results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "BELLMAN_FORD")
            os.makedirs(results_dir, exist_ok=True)
            output_path = standardize_path(os.path.join(results_dir, file_name))
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"[OK] Visualisation Bellman-Ford sauvegardée dans: {output_path}")
            plt.close()
        return fig
