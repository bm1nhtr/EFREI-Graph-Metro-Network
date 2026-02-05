"""
Algorithme de Kruskal - Arbre couvrant de poids minimum (MST)
Sur le graphe non orienté du réseau métro (poids = temps de trajet).
"""

from algorithms.utils import standardize_path, LAYOUT_METRO, LAYOUT_METRO_GUI
import os
import matplotlib.pyplot as plt
import networkx as nx


class Kruskal:
    def __init__(self, graph_data):
        # Stocke les données du graphe (arêtes + poids)
        self.graph_data = graph_data

    def get_edges_with_weights(self):
        """Retourne la liste des arêtes avec poids (sans doublons)."""
        # Liste finale des arêtes uniques
        edges = []
        # Ensemble pour éviter les doublons dans un graphe non orienté
        seen = set()

        # Parcourir les données brutes du graphe
        for edge in self.graph_data:
            u, v, w = int(edge[0]), int(edge[1]), float(edge[2])
            # Normaliser l’arête pour éviter (u,v) et (v,u)
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                edges.append((u, v, w))

        return edges

    # --------- UNION FIND ---------
    def find(self, parent, node):
        # Trouve la racine d’un nœud avec compression de chemin
        if parent[node] != node:
            parent[node] = self.find(parent, parent[node])
        return parent[node]

    def union(self, parent, rank, x, y):
        # Fusionne deux composantes en respectant la hauteur des arbres
        root_x = self.find(parent, x)
        root_y = self.find(parent, y)

        if root_x != root_y:
            if rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
            elif rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            else:
                parent[root_y] = root_x
                rank[root_x] += 1

    # --------- KRUSKAL ---------
    def kruskal_mst(self, start_node: int):
        """
        Calcule l’arbre couvrant minimum avec l’algorithme de Kruskal.
        """
        # Récupérer toutes les arêtes uniques du graphe
        edges = self.get_edges_with_weights()

        # Trier les arêtes par poids croissant
        edges.sort(key=lambda x: x[2])

        # Extraire l’ensemble des nœuds du graphe
        nodes = set()
        for u, v, _ in edges:
            nodes.add(u)
            nodes.add(v)

        # Initialiser Union-Find (chaque nœud est sa propre racine)
        parent = {node: node for node in nodes}
        rank = {node: 0 for node in nodes}

        # Stocker les arêtes du MST
        mst_edges = []
        # Stocker le poids total du MST
        total_weight = 0.0

        # Parcourir les arêtes dans l’ordre croissant
        for u, v, w in edges:
            # Ajouter l’arête si elle ne crée pas de cycle
            if self.find(parent, u) != self.find(parent, v):
                self.union(parent, rank, u, v)
                mst_edges.append((u, v, w))
                total_weight += w

        return mst_edges, total_weight

    def sauvegarder_resultats(self, mst_edges, total_weight, file_name="kruskal_result.txt"):
        """Sauvegarde les arêtes du MST et le poids total dans un fichier texte."""
        # Créer le dossier de résultats s’il n’existe pas
        results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "KRUSKAL")
        os.makedirs(results_dir, exist_ok=True)
        output_path = standardize_path(os.path.join(results_dir, file_name))

        # Écrire les résultats dans un fichier texte
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Résultats Algorithme de Kruskal - Arbre couvrant minimum (MST)\n")
            f.write(f"# Poids total du MST: {total_weight}\n")
            f.write(f"# Nombre d'arêtes: {len(mst_edges)}\n\n")
            f.write("# Arêtes (station_u station_v poids)\n")
            for u, v, w in mst_edges:
                f.write(f"{u} {v} {w}\n")

        print(f"[OK] Résultats Kruskal sauvegardés dans: {output_path}")

    def visualiser_mst(self, mst_edges, total_weight, start_node, file_name="kruskal_visualization.png", fig=None):
        
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

        # Arêtes du MST en vert
        mst_edge_set = set()
        for u, v, _ in mst_edges:
            mst_edge_set.add((u, v))
            mst_edge_set.add((v, u))
        edgelist_mst = [(u, v) for u, v in G.edges()
                        if (u, v) in mst_edge_set or (v, u) in mst_edge_set]
        nx.draw_networkx_edges(G, pos, edgelist=edgelist_mst,
                            edge_color="green", width=5, alpha=0.9, ax=ax)

        # Poids sur les arêtes (comme Prim)
        edge_list = list(G.edges())
        edge_labels = {(u, v): str(G[u][v]["weight"]) for u, v in edge_list}
        for edges_sub, label_pos in [(edge_list[0::3], 0.25),
                                    (edge_list[1::3], 0.5),
                                    (edge_list[2::3], 0.75)]:
            sub_labels = {e: edge_labels[e] for e in edges_sub if e in edge_labels}
            if sub_labels:
                nx.draw_networkx_edge_labels(
                    G, pos, sub_labels, font_size=10, label_pos=label_pos, ax=ax,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85)
                )

        # Nœuds du MST
        mst_nodes = set()
        for u, v, _ in mst_edges:
            mst_nodes.add(u)
            mst_nodes.add(v)
        nx.draw_networkx_nodes(G, pos, nodelist=list(mst_nodes),
                            node_color="seagreen", node_size=600,
                            alpha=0.9, ax=ax, edgecolors="black", linewidths=2)

        # Racine en rouge (pour cohérence avec Prim)
        nx.draw_networkx_nodes(G, pos, nodelist=[start_node],
                            node_color="red", node_size=800,
                            alpha=1.0, ax=ax, edgecolors="darkred", linewidths=4)

        labels = {node: str(node) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)

        ax.set_title(
            f"MST (Kruskal) - Racine: Station {start_node}\n"
            f"Poids total: {total_weight} min | Arêtes: {len(mst_edges)}",
            fontsize=16, fontweight="bold", pad=20
        )
        ax.axis("off")

        # Légende (comme Prim, adaptée)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="red", label=f"Racine ({start_node})"),
            Patch(facecolor="seagreen", label="Stations dans le MST"),
            Patch(facecolor="lightgray", label="Autres stations"),
            Patch(facecolor="green", edgecolor="green", label="Arêtes du MST (Kruskal)"),
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
            results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "KRUSKAL")
            os.makedirs(results_dir, exist_ok=True)
            output_path = standardize_path(os.path.join(results_dir, file_name))
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"[OK] Visualisation Kruskal sauvegardée dans: {output_path}")
            plt.close()
        return fig