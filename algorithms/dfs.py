"""
Parcours en profondeur (DFS) sur le graphe du réseau métro.

Complexité : O(V + E). Espace : O(V).
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


class DFS:
    """Parcours en profondeur (DFS) sur le graphe du réseau métro."""

    def __init__(self, graph_data):
        """Initialise avec les données du graphe (tableau d'arêtes [u, v, poids])."""
        self.graph_data = graph_data

    def get_neighbors(self, node: int):
        # Retourne l’ensemble des voisins d’un nœud donné
        neighbors = set()
        node = int(node)

        # Parcourir toutes les arêtes pour trouver les voisins
        for edge in self.graph_data:
            source, target, _ = int(edge[0]), int(edge[1]), edge[2]
            if source == node:
                neighbors.add(target)
            elif target == node:
                neighbors.add(source)

        return neighbors

    def parcourir_dfs(self, start_node: int):
        """Parcours (DFS). Complexité O(V+E). Retourne (ordre_visite, parent)."""
        # Initialiser le nœud de départ
        start_node = int(start_node)

        # Ensemble des nœuds déjà visités
        visited = set()
        # Pile pour le parcours DFS
        stack = [start_node]
        # Ordre de visite des nœuds
        result = []
        # Dictionnaire parent → enfant pour reconstruire l’arbre DFS
        parent = {}

        # Parcours tant que la pile n’est pas vide
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                result.append(node)

                # Récupérer les voisins dans un ordre déterministe
                neighbors = sorted(self.get_neighbors(node), reverse=True)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        parent[neighbor] = node
                        stack.append(neighbor)

        return result, parent

    def parcourir_dfs_steps(self, start_node: int):
        """DFS étape par étape : yield un dict par nœud visité (pour visualisation web). Même complexité O(V+E)."""
        start_node = int(start_node)
        visited = set()
        stack = [start_node]
        result = []
        parent = {}
        step_index = 0
        yield {
            "step_index": step_index,
            "description": f"Départ : pile = [{start_node}], visités = []",
            "visited": [],
            "stack": [start_node],
            "current_node": None,
        }
        step_index += 1
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                result.append(node)
                neighbors = sorted(self.get_neighbors(node), reverse=True)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        parent[neighbor] = node
                        stack.append(neighbor)
                yield {
                    "step_index": step_index,
                    "description": f"Nœud {node} visité. Pile (sommet en dernier) = {list(stack)[-10:] if len(stack) > 10 else list(stack)}",
                    "visited": list(result),
                    "stack": list(stack),
                    "current_node": node,
                }
                step_index += 1
        yield {
            "step_index": step_index,
            "description": f"Fin. Ordre de visite : {' → '.join(map(str, result))}",
            "visited": list(result),
            "stack": [],
            "current_node": None,
        }

    def sauvegarder_resultats(self, parcours, file_name="dfs_result.txt"):
        """Sauvegarde l’ordre de visite du DFS dans un fichier texte."""
        # Créer le dossier de résultats s’il n’existe pas
        results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "DFS")
        os.makedirs(results_dir, exist_ok=True)
        output_path = standardize_path(os.path.join(results_dir, file_name))

        # Écrire les résultats du DFS dans un fichier texte
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Résultats du parcours DFS\n")
            f.write("# Format: Liste des nœuds visités dans l'ordre\n")
            f.write(f"# Nombre de nœuds visités: {len(parcours)}\n\n")
            f.write(" -> ".join(map(str, parcours)))
            f.write("\n")

        print(f"[OK] Résultats DFS sauvegardés dans: {output_path}")

    def visualiser_parcours(
        self, parcours, parent, start_node, file_name="dfs_visualization.png", fig=None
    ):
        """
        Visualise le parcours DFS sur le graphe (couleurs par ordre de visite).

        Si fig est fourni (mode GUI), dessine dessus sans sauvegarder.
        Sinon crée une figure, sauvegarde dans results/DFS/ et ferme.

        Args:
            parcours: Liste des nœuds dans l'ordre de visite.
            parent: Dict nœud -> parent (arbre DFS).
            start_node: Nœud de départ.
            file_name: Nom du fichier image si non interactif.
            fig: Figure matplotlib optionnelle (mode GUI).

        Returns:
            Figure matplotlib (pour réutilisation en GUI).
        """
        interactive = fig is not None
        if not interactive:
            fig, ax = plt.subplots(figsize=(16, 12))
        else:
            fig.clear()
            ax = fig.add_subplot(111)

        # Construire le graphe NetworkX à partir des données
        G = nx.Graph()
        for edge in self.graph_data:
            source, target, weight = edge
            G.add_edge(source, target, weight=weight)

        # Calculer une disposition stable des nœuds
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Dessiner tous les nœuds du graphe en gris clair
        nx.draw_networkx_nodes(G, pos, node_color="lightgray", node_size=400, alpha=0.6, ax=ax)

        # Dessiner toutes les arêtes avec une couleur liée au poids
        edges = G.edges()
        weights = [G[u][v]["weight"] for u, v in edges]
        edge_colors = plt.cm.Reds([w / 4.0 for w in weights])
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2.5, alpha=0.6, ax=ax)

        # Construire les arêtes de l’arbre DFS à partir du dictionnaire parent
        dfs_tree_edges = []
        for child, par in parent.items():
            dfs_tree_edges.append((par, child))

        # Dessiner les arêtes utilisées par le DFS en bleu épais
        if dfs_tree_edges:
            nx.draw_networkx_edges(
                G, pos, edgelist=dfs_tree_edges, edge_color="blue", width=5, alpha=0.9, ax=ax
            )

        # Afficher les poids sur les arêtes en limitant les chevauchements
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

        # Associer chaque nœud à son ordre de visite
        visit_order = {node: idx for idx, node in enumerate(parcours)}

        # Préparer les couleurs des nœuds visités selon l’ordre DFS
        visited_nodes = []
        visited_colors = []
        for node in parcours:
            visited_nodes.append(node)
            order_ratio = visit_order[node] / max(len(parcours) - 1, 1)
            visited_colors.append(plt.cm.viridis(order_ratio))

        # Dessiner les nœuds visités avec un gradient de couleur
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=visited_nodes,
            node_color=visited_colors,
            node_size=600,
            alpha=0.9,
            ax=ax,
            edgecolors="black",
        )

        # Mettre en évidence le nœud de départ en rouge
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[start_node],
            node_color="red",
            node_size=800,
            alpha=1.0,
            ax=ax,
            edgecolors="darkred",
        )

        # Afficher les labels des nœuds
        labels = {node: str(node) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)

        # Ajouter un titre descriptif
        ax.set_title(
            f"Parcours DFS - Départ: Station {start_node}\n"
            f"Ordre de visite: {len(parcours)} stations | Arêtes utilisées: {len(dfs_tree_edges)}",
            fontsize=16,
            fontweight="bold",
        )
        ax.axis("off")

        # Créer la légende explicative
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="red", label=f"Station de départ ({start_node})"),
            Patch(facecolor="lightblue", label="Stations visitées (gradient = ordre DFS)"),
            Patch(facecolor="lightgray", label="Stations non visitées"),
            Patch(facecolor="blue", edgecolor="blue", label="Arêtes de l'arbre DFS"),
        ]

        # Positionner la légende selon le mode
        if interactive:
            ax.legend(
                handles=legend_elements, loc="upper left", fontsize=10, bbox_to_anchor=(1.02, 1)
            )
        else:
            ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

        # Ajouter une barre de couleur pour l’ordre de visite
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=len(parcours) - 1)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)
        cbar.set_label("Ordre de visite DFS", rotation=270, labelpad=20)

        # Ajuster la mise en page
        if interactive:
            fig.subplots_adjust(**LAYOUT_METRO_GUI)
        else:
            fig.subplots_adjust(**LAYOUT_METRO)
            plt.tight_layout()

        # Sauvegarder l’image si hors GUI
        if not interactive:
            results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "DFS")
            os.makedirs(results_dir, exist_ok=True)
            output_path = standardize_path(os.path.join(results_dir, file_name))
            plt.savefig(output_path, dpi=EXPORT_DPI, bbox_inches="tight", **SAVEFIG_PNG_OPTIONS)
            print(f"[OK] Visualisation DFS sauvegardée dans: {output_path}")
            plt.close()

        return fig

    def visualiser_arbre_dfs(
        self, parcours, parent, start_node, file_name="dfs_tree_visualization.png", fig=None
    ):
        """Visualiser l'arbre DFS en hiérarchie (structure arborescente).
        Si fig est fourni (GUI), dessine dessus et ne sauvegarde pas.

        Args:
            parcours (list): Liste des nœuds visités dans l'ordre de la visite DFS.
            parent (dict): Dictionnaire {enfant: parent} issu du DFS.
            start_node (int): Nœud de départ du parcours DFS.
            file_name (str): Nom du fichier pour sauvegarder la visualisation.
        """
        # Créer un graphe pour l'arbre DFS
        G_tree = nx.DiGraph()  # Graphe orienté pour l'arbre

        # Construire l'arbre DFS à partir du dictionnaire parent
        for child, par in parent.items():
            G_tree.add_edge(par, child)

        # Calculer les niveaux (profondeur) dans l'arbre DFS
        levels = {start_node: 0}
        for node in parcours:
            if node in parent:
                levels[node] = levels[parent[node]] + 1

        # Créer une disposition hiérarchique (arbre)
        pos = {}
        max_level = max(levels.values()) if levels else 0

        # Organiser les nœuds par niveau
        nodes_by_level = {}
        for node, level in levels.items():
            if level not in nodes_by_level:
                nodes_by_level[level] = []
            nodes_by_level[level].append(node)

        # Positionner les nœuds : x selon leur position dans le niveau, y selon le niveau
        for level in range(max_level + 1):
            if level in nodes_by_level:
                nodes_in_level = nodes_by_level[level]
                num_nodes = len(nodes_in_level)
                y_pos = max_level - level  # Racine en haut
                for idx, node in enumerate(nodes_in_level):
                    x_pos = idx - (num_nodes - 1) / 2
                    pos[node] = (x_pos, y_pos)

        interactive = fig is not None
        if not interactive:
            fig, ax = plt.subplots(figsize=(18, 12))
        else:
            fig.clear()
            ax = fig.add_subplot(111)

        # Dessiner les arêtes de l'arbre
        nx.draw_networkx_edges(
            G_tree,
            pos,
            edge_color="purple",
            width=3,
            alpha=0.8,
            ax=ax,
            arrows=True,
            arrowsize=20,
            arrowstyle="->",
            connectionstyle="arc3,rad=0.1",
        )

        # Poids sur les arêtes (pour vérification manuelle)
        weight_map = {}
        for edge in self.graph_data:
            u, v, w = int(edge[0]), int(edge[1]), int(edge[2])
            weight_map[(min(u, v), max(u, v))] = w

        edge_list_tree = list(G_tree.edges())
        edge_labels_tree = {
            (u, v): str(weight_map.get((min(u, v), max(u, v)), "")) for u, v in edge_list_tree
        }

        for edges_sub, label_pos in [
            (edge_list_tree[0::3], 0.25),
            (edge_list_tree[1::3], 0.5),
            (edge_list_tree[2::3], 0.75),
        ]:
            sub_labels = {e: edge_labels_tree[e] for e in edges_sub if e in edge_labels_tree}
            if sub_labels:
                nx.draw_networkx_edge_labels(
                    G_tree,
                    pos,
                    sub_labels,
                    font_size=10,
                    label_pos=label_pos,
                    ax=ax,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85),
                )

        # Ordre de visite (pour les couleurs)
        visit_order = {node: idx for idx, node in enumerate(parcours)}

        node_colors = []
        node_sizes = []
        for node in G_tree.nodes():
            if node in visit_order:
                order_ratio = visit_order[node] / max(len(parcours) - 1, 1)
                node_colors.append(plt.cm.viridis(order_ratio))
                node_sizes.append(700)
            else:
                node_colors.append("lightgray")
                node_sizes.append(500)

        # Dessiner les nœuds
        nx.draw_networkx_nodes(
            G_tree,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            ax=ax,
            edgecolors="black",
            linewidths=2,
        )

        # Mettre en évidence la racine
        nx.draw_networkx_nodes(
            G_tree,
            pos,
            nodelist=[start_node],
            node_color="red",
            node_size=900,
            alpha=1.0,
            ax=ax,
            edgecolors="darkred",
            linewidths=4,
        )

        # Labels
        labels = {node: str(node) for node in G_tree.nodes()}
        nx.draw_networkx_labels(G_tree, pos, labels, font_size=11, font_weight="bold", ax=ax)

        # Lignes de niveau
        for level in range(max_level + 1):
            y_pos = max_level - level
            ax.axhline(y=y_pos, color="gray", linestyle="--", alpha=0.3, linewidth=1)
            ax.text(
                -max([len(nodes_by_level.get(lv, [])) for lv in range(max_level + 1)]) / 2 - 0.5,
                y_pos,
                f"Niveau {level}",
                fontsize=10,
                verticalalignment="center",
                color="gray",
            )

        # Titre
        title = f"Arbre DFS Hiérarchique - Départ: Station {start_node}\n"
        title += f"Nombre de nœuds: {len(parcours)} | Profondeur: {max_level} niveaux"
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.axis("off")
        ax.set_aspect("equal")

        # Légende
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="red", label=f"Racine (Station {start_node})"),
            Patch(facecolor="lightblue", label="Nœuds (gradient = ordre de visite)"),
            Patch(facecolor="purple", edgecolor="purple", label="Arêtes de l'arbre DFS"),
        ]
        if interactive:
            ax.legend(
                handles=legend_elements, loc="upper left", fontsize=10, bbox_to_anchor=(1.02, 1)
            )
        else:
            ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

        # Barre de couleur
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=len(parcours) - 1)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)
        cbar.set_label("Ordre de visite DFS", rotation=270, labelpad=20)

        if interactive:
            fig.subplots_adjust(**LAYOUT_ARBRE_GUI)
        else:
            fig.subplots_adjust(**LAYOUT_ARBRE)
            plt.tight_layout()
        if not interactive:
            results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "DFS")
            os.makedirs(results_dir, exist_ok=True)
            output_path = standardize_path(os.path.join(results_dir, file_name))
            plt.savefig(output_path, dpi=EXPORT_DPI, bbox_inches="tight", **SAVEFIG_PNG_OPTIONS)
            print(f"[OK] Visualisation arbre DFS sauvegardée dans: {output_path}")
            plt.close()

        return fig
