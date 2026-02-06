import os

import matplotlib.pyplot as plt
import networkx as nx

from algorithms.utils import (
    LAYOUT_ARBRE,
    LAYOUT_ARBRE_GUI,
    LAYOUT_METRO,
    LAYOUT_METRO_GUI,
    standardize_path,
)


class BFS:
    """Parcours en largeur (BFS) sur le graphe du réseau métro."""

    def __init__(self, graph_data):
        """Initialise avec les données du graphe (tableau d'arêtes [u, v, poids])."""
        self.graph_data = graph_data

    def get_neighbors(self, node: int):
        """Retourne les voisins d'un nœud.
        Args:
            node (int): Nœud dont on veut trouver les voisins.

        Returns:
                set: Ensemble des voisins du nœud.
        """
        neighbors = set()
        node = int(node)
        # Parcourir les arêtes pour trouver les voisins
        for edge in self.graph_data:
            source, target, _ = int(edge[0]), int(edge[1]), edge[2]
            # Si le nœud est la source, ajouter la cible comme voisin
            if source == node:
                neighbors.add(target)
            # Si le nœud est la cible, ajouter la source comme voisin
            elif target == node:
                neighbors.add(source)
        return neighbors

    def parcourir_bfs(self, start_node: int):
        """
        Effectuer une recherche en largeur (BFS) à partir d'un nœud de départ donné.
        Args:
            start_node (int): Le nœud de départ pour la recherche BFS.
        Returns:
            list: Liste des nœuds visités dans l'ordre de la visite BFS.
        """
        start_node = int(start_node)
        visited = set()
        queue = [start_node]
        result = []

        while queue:
            # Extraire le premier nœud de la file d'attente
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                result.append(int(node))
                neighbors = self.get_neighbors(node)
                # Ajouter les voisins non visités à la file d'attente
                queue.extend(neighbors - visited)

        return result

    def parcourir_bfs_steps(self, start_node: int):
        """BFS étape par étape : yield un dict par nœud visité (pour visualisation web)."""
        start_node = int(start_node)
        visited = set()
        queue = [start_node]
        result = []
        step_index = 0
        # Step 0: initialisation
        yield {
            "step_index": step_index,
            "description": f"Départ : file = [{start_node}], visités = []",
            "visited": [],
            "queue": [start_node],
            "current_node": None,
        }
        step_index += 1
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                result.append(int(node))
                neighbors = self.get_neighbors(node)
                queue.extend(neighbors - visited)
                yield {
                    "step_index": step_index,
                    "description": f"Nœud {node} visité. Voisins ajoutés à la file : {sorted(neighbors - visited) or '(aucun)'}. File = {list(queue)}",
                    "visited": list(result),
                    "queue": list(queue),
                    "current_node": node,
                }
                step_index += 1
        # Step final
        yield {
            "step_index": step_index,
            "description": f"Fin. Ordre de visite : {' → '.join(map(str, result))}",
            "visited": list(result),
            "queue": [],
            "current_node": None,
        }

    def sauvegarder_resultats(self, parcours, file_name="bfs_result.txt"):
        """Sauvegarder les résultats du parcours BFS dans un fichier texte.

        Args:
            parcours (list): Liste des nœuds visités dans l'ordre de la visite BFS.
            file_name (str): Nom du fichier pour sauvegarder les résultats.
        """
        results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "BFS")
        os.makedirs(results_dir, exist_ok=True)
        output_path = standardize_path(os.path.join(results_dir, file_name))

        # Sauvegarder les résultats dans un fichier texte
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Résultats du parcours BFS\n")
            f.write("# Format: Liste des nœuds visités dans l'ordre\n")
            f.write(f"# Nombre de nœuds visités: {len(parcours)}\n\n")
            f.write(" -> ".join(map(str, parcours)))
            f.write("\n")

        print(f"[OK] Résultats sauvegardés dans: {output_path}")

    def _reconstruire_arbre_bfs(self, start_node):
        """Reconstruire l'arbre BFS pour trouver les vraies arêtes utilisées.

        Args:
            parcours (list): Liste des nœuds visités dans l'ordre.
            start_node (int): Nœud de départ.

        Returns:
            set: Ensemble des arêtes (tuples) utilisées dans l'arbre BFS.
        """
        bfs_tree_edges = set()
        visited = set()
        queue = [start_node]
        parent = {}  # parent[node] = nœud parent dans l'arbre BFS

        # Parcourir le graphe pour reconstruire l'arbre BFS
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                if node != start_node:
                    # Ajouter l'arête vers le parent
                    bfs_tree_edges.add((parent[node], node))
                    bfs_tree_edges.add((node, parent[node]))  # Non orienté

                # Ajouter les voisins non visités à la file d'attente
                neighbors = self.get_neighbors(node)
                for neighbor in neighbors:
                    if neighbor not in visited and neighbor not in queue:
                        parent[neighbor] = node
                        queue.append(neighbor)

        return bfs_tree_edges

    def visualiser_parcours(
        self, parcours, start_node, file_name="bfs_visualization.png", fig=None
    ):
        """Visualiser le parcours BFS sur le graphe du réseau métro.
        Si fig est fourni (GUI), dessine dessus et ne sauvegarde pas."""
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

        # D'abord, dessiner le graphe de base (comme visualize_metro)
        # Dessiner tous les nœuds en gris clair
        nx.draw_networkx_nodes(G, pos, node_color="lightgray", node_size=400, alpha=0.6, ax=ax)

        # Dessiner toutes les arêtes - rendre plus visibles
        edges = G.edges()
        weights = [G[u][v]["weight"] for u, v in edges]
        edge_colors = plt.cm.Reds([w / 4.0 for w in weights])
        # Rendre les arêtes non parcourues plus visibles
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2.5, alpha=0.6, ax=ax)

        # Reconstruire l'arbre BFS pour trouver les vraies arêtes utilisées
        bfs_tree_edges = self._reconstruire_arbre_bfs(start_node)

        # Dessiner les arêtes de l'arbre BFS en bleu épais
        if bfs_tree_edges:
            nx.draw_networkx_edges(
                G, pos, edgelist=list(bfs_tree_edges), edge_color="blue", width=5, alpha=0.9, ax=ax
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

        # Créer un dictionnaire pour l'ordre de visite (pour les couleurs)
        visit_order = {node: idx for idx, node in enumerate(parcours)}

        # Dessiner les nœuds visités avec gradient de couleur
        visited_nodes = []
        visited_colors = []
        visited_sizes = []

        for node in parcours:
            visited_nodes.append(node)
            order_ratio = visit_order[node] / max(len(parcours) - 1, 1)
            visited_colors.append(plt.cm.viridis(order_ratio))
            visited_sizes.append(600)

        # Dessiner les nœuds visités par-dessus
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=visited_nodes,
            node_color=visited_colors,
            node_size=visited_sizes,
            alpha=0.9,
            ax=ax,
            edgecolors="black",
            linewidths=2,
        )

        # Mettre en évidence le nœud de départ (rouge pour contraster avec les bleus/verts)
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

        # Dessiner les labels pour tous les nœuds
        labels = {node: str(node) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)

        # Ajouter titre et informations
        title = f"Parcours BFS - Départ: Station {start_node}\n"
        title += f"Ordre de visite: {len(parcours)} stations | Arêtes utilisées: {len(bfs_tree_edges) // 2}"
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.axis("off")

        # Ajouter légende (en GUI : à l'extérieur pour ne pas chevaucher)
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="red", label=f"Station de départ ({start_node})"),
            Patch(facecolor="lightblue", label="Stations visitées (gradient = ordre)"),
            Patch(facecolor="lightgray", label="Stations non visitées"),
            Patch(facecolor="blue", edgecolor="blue", label="Arêtes de l'arbre BFS"),
        ]
        if interactive:
            ax.legend(
                handles=legend_elements, loc="upper left", fontsize=10, bbox_to_anchor=(1.02, 1)
            )
        else:
            ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

        # Barre de couleur pour l'ordre de visite
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=len(parcours) - 1)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)
        cbar.set_label("Ordre de visite BFS", rotation=270, labelpad=20)

        if interactive:
            fig.subplots_adjust(**LAYOUT_METRO_GUI)
        else:
            fig.subplots_adjust(**LAYOUT_METRO)
            plt.tight_layout()
        if not interactive:
            results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "BFS")
            os.makedirs(results_dir, exist_ok=True)
            output_path = standardize_path(os.path.join(results_dir, file_name))
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"[OK] Visualisation BFS sauvegardée dans: {output_path}")
            plt.close()
        return fig

    def _calculer_niveaux_bfs(self, parcours, start_node):
        """Calculer les niveaux (profondeur) de chaque nœud dans l'arbre BFS.

        Args:
            parcours (list): Liste des nœuds visités dans l'ordre.
            start_node (int): Nœud de départ.

        Returns:
            dict: Dictionnaire {node: niveau} où niveau 0 = racine
        """
        levels = {start_node: 0}
        visited = set()
        queue = [start_node]

        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                current_level = levels[node]

                neighbors = self.get_neighbors(node)
                for neighbor in neighbors:
                    if neighbor not in visited and neighbor not in queue:
                        levels[neighbor] = current_level + 1
                        queue.append(neighbor)

        return levels

    def visualiser_arbre_bfs(
        self, parcours, start_node, file_name="bfs_tree_visualization.png", fig=None
    ):
        """Visualiser l'arbre BFS en hiérarchie (structure arborescente).
        Si fig est fourni (GUI), dessine dessus et ne sauvegarde pas.

        Args:
            parcours (list): Liste des nœuds visités dans l'ordre de la visite BFS.
            start_node (int): Nœud de départ du parcours BFS.
            file_name (str): Nom du fichier pour sauvegarder la visualisation.
        """
        # Créer un graphe pour l'arbre BFS
        G_tree = nx.DiGraph()  # Graphe orienté pour l'arbre

        # Reconstruire l'arbre BFS
        visited = set()
        queue = [start_node]
        parent = {}
        bfs_tree_edges = []

        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                if node != start_node:
                    # Ajouter l'arête parent -> enfant dans l'arbre
                    G_tree.add_edge(parent[node], node)
                    bfs_tree_edges.append((parent[node], node))

                neighbors = self.get_neighbors(node)
                for neighbor in neighbors:
                    if neighbor not in visited and neighbor not in queue:
                        parent[neighbor] = node
                        queue.append(neighbor)

        # Calculer les niveaux
        levels = self._calculer_niveaux_bfs(parcours, start_node)

        # Créer une disposition hiérarchique (arbre)
        # Utiliser un layout en arbre avec les niveaux
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
                y_pos = max_level - level  # Inverser pour que la racine soit en haut
                for idx, node in enumerate(nodes_in_level):
                    x_pos = idx - (num_nodes - 1) / 2
                    pos[node] = (x_pos, y_pos)

        interactive = fig is not None
        if not interactive:
            fig, ax = plt.subplots(figsize=(18, 12))
        else:
            fig.clear()
            ax = fig.add_subplot(111)

        # Dessiner les arêtes de l'arbre (orientées de haut en bas)
        nx.draw_networkx_edges(
            G_tree,
            pos,
            edge_color="blue",
            width=3,
            alpha=0.8,
            ax=ax,
            arrows=True,
            arrowsize=20,
            arrowstyle="->",
            connectionstyle="arc3,rad=0.1",
        )

        # Poids sur les arêtes (pour vérification manuelle) — répartis pour limiter les chevauchements
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

        # Créer un dictionnaire pour l'ordre de visite (pour les couleurs)
        visit_order = {node: idx for idx, node in enumerate(parcours)}

        # Dessiner les nœuds avec gradient de couleur selon l'ordre de visite
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

        # Mettre en évidence la racine (rouge pour contraster)
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

        # Dessiner les labels pour tous les nœuds
        labels = {node: str(node) for node in G_tree.nodes()}
        nx.draw_networkx_labels(G_tree, pos, labels, font_size=11, font_weight="bold", ax=ax)

        # Ajouter des lignes de niveau (optionnel)
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

        # Ajouter titre
        title = f"Arbre BFS Hiérarchique - Départ: Station {start_node}\n"
        title += f"Nombre de nœuds: {len(parcours)} | Profondeur: {max_level} niveaux"
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.axis("off")
        ax.set_aspect("equal")

        # Ajouter légende
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="red", label=f"Racine (Station {start_node})"),
            Patch(facecolor="lightblue", label="Nœuds (gradient = ordre de visite)"),
            Patch(facecolor="blue", edgecolor="blue", label="Arêtes de l'arbre (parent → enfant)"),
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
        cbar.set_label("Ordre de visite BFS", rotation=270, labelpad=20)

        if interactive:
            fig.subplots_adjust(**LAYOUT_ARBRE_GUI)
        else:
            fig.subplots_adjust(**LAYOUT_ARBRE)
            plt.tight_layout()
        if not interactive:
            results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "BFS")
            os.makedirs(results_dir, exist_ok=True)
            output_path = standardize_path(os.path.join(results_dir, file_name))
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"[OK] Visualisation arbre BFS sauvegardée dans: {output_path}")
            plt.close()
        return fig
