from algorithms.utils import load_graph_data
from algorithms.utils import standardize_path
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class BFS:
    def __init__(self,graph_data):
        self.graph_data = graph_data
    
    def get_neighbors(self, node:int):
            """Retourne les voisins d'un nœud.
            Args:
                node (int): Nœud dont on veut trouver les voisins.
            
            Returns:
                    set: Ensemble des voisins du nœud.
            """
            neighbors = set()
            # Parcourir les arêtes pour trouver les voisins
            for edge in self.graph_data:
                source, target, _ = edge
                # Si le nœud est la source, ajouter la cible comme voisin
                if source == node:
                    neighbors.add(target)
                # Si le nœud est la cible, ajouter la source comme voisin
                elif target == node:
                    neighbors.add(source)
            return neighbors

    def parcourir_bfs(self, start_node:int):
        """
        Effectuer une recherche en largeur (BFS) à partir d'un nœud de départ donné.
        Args:
            start_node (int): Le nœud de départ pour la recherche BFS.
        Returns:
            list: Liste des nœuds visités dans l'ordre de la visite BFS.
        """
        visited = set()
        queue = [start_node]
        result = []
        
        while queue:
            # Extraire le premier nœud de la file d'attente
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                result.append(node)
                neighbors = self.get_neighbors(node)
                # Ajouter les voisins non visités à la file d'attente
                queue.extend(neighbors - visited)

        return result
    
    def sauvegarder_resultats(self, parcours, file_name='bfs_result.txt'):
        """Sauvegarder les résultats du parcours BFS dans un fichier texte.
        
        Args:
            parcours (list): Liste des nœuds visités dans l'ordre de la visite BFS.
            file_name (str): Nom du fichier pour sauvegarder les résultats.
        """
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        output_path = standardize_path(os.path.join(results_dir, file_name))
        
        # Sauvegarder les résultats dans un fichier texte
        with open(output_path, 'w', encoding='utf-8') as f:
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
    
    def visualiser_parcours(self, parcours, start_node, file_name='bfs_visualization.png'):
        """Visualiser le parcours BFS sur le graphe du réseau métro.
        Utilise la visualisation générale comme base et dessine par-dessus.
        
        Args:
            parcours (list): Liste des nœuds visités dans l'ordre de la visite BFS.
            start_node (int): Nœud de départ du parcours BFS.
            file_name (str): Nom du fichier pour sauvegarder la visualisation.
        """
        # Importer la fonction de visualisation générale
        from algorithms.visualize_metro import load_graph_data
        
        # Créer un graphe non orienté (même code que visualize_metro)
        G = nx.Graph()
        for edge in self.graph_data:
            source, target, weight = edge
            G.add_edge(source, target, weight=weight)
        
        # Utiliser la même disposition que la visualisation générale
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # D'abord, dessiner le graphe de base (comme visualize_metro)
        # Dessiner tous les nœuds en gris clair
        nx.draw_networkx_nodes(G, pos, node_color='lightgray', 
                              node_size=400, alpha=0.6, ax=ax)
        
        # Dessiner toutes les arêtes - rendre plus visibles
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        edge_colors = plt.cm.Reds([w/4.0 for w in weights])
        # Rendre les arêtes non parcourues plus visibles
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, 
                              width=2.5, alpha=0.6, ax=ax)
        
        # Reconstruire l'arbre BFS pour trouver les vraies arêtes utilisées
        bfs_tree_edges = self._reconstruire_arbre_bfs(start_node)
        
        # Dessiner les arêtes de l'arbre BFS en bleu épais
        if bfs_tree_edges:
            nx.draw_networkx_edges(G, pos, edgelist=list(bfs_tree_edges), 
                                  edge_color='blue', width=5, alpha=0.9, ax=ax)
        
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
        nx.draw_networkx_nodes(G, pos, nodelist=visited_nodes, 
                              node_color=visited_colors, 
                              node_size=visited_sizes, 
                              alpha=0.9, ax=ax, edgecolors='black', linewidths=2)
        
        # Mettre en évidence le nœud de départ
        nx.draw_networkx_nodes(G, pos, nodelist=[start_node], 
                              node_color='green', node_size=800, 
                              alpha=1.0, ax=ax, edgecolors='darkgreen', linewidths=4)
        
        # Dessiner les labels pour tous les nœuds
        labels = {node: str(node) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)
        
        # Label spécial pour le nœud de départ
        start_label = {start_node: f'DÉPART\n{start_node}'}
        nx.draw_networkx_labels(G, pos, start_label, font_size=12, 
                               font_weight='bold', font_color='darkgreen', ax=ax)
        
        # Ajouter titre et informations
        title = f'Parcours BFS - Départ: Station {start_node}\n'
        title += f'Ordre de visite: {len(parcours)} stations | Arêtes utilisées: {len(bfs_tree_edges)//2}'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Ajouter légende
        from matplotlib.patches import Patch
        
        legend_elements = [
            Patch(facecolor='green', label=f'Station de départ ({start_node})'),
            Patch(facecolor='lightblue', label='Stations visitées (gradient = ordre)'),
            Patch(facecolor='lightgray', label='Stations non visitées'),
            Patch(facecolor='blue', edgecolor='blue', label='Arêtes de l\'arbre BFS'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Ajouter une barre de couleur pour montrer l'ordre de visite
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                   norm=plt.Normalize(vmin=0, vmax=len(parcours)-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', 
                           fraction=0.02, pad=0.04)
        cbar.set_label('Ordre de visite BFS', rotation=270, labelpad=20)
        
        plt.tight_layout()
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        output_path = standardize_path(os.path.join(results_dir, file_name))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Visualisation BFS sauvegardée dans: {output_path}")
        plt.close()
    
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
    
    def visualiser_arbre_bfs(self, parcours, start_node, file_name='bfs_tree_visualization.png'):
        """Visualiser l'arbre BFS en hiérarchie (structure arborescente).
        Les nœuds sont organisés par niveaux qui descendent.
        
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
                    x_pos = idx - (num_nodes - 1) / 2  # Centrer les nœuds
                    pos[node] = (x_pos, y_pos)
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=(18, 12))
        
        # Dessiner les arêtes de l'arbre (orientées de haut en bas)
        nx.draw_networkx_edges(G_tree, pos, edge_color='blue', 
                              width=3, alpha=0.8, ax=ax, 
                              arrows=True, arrowsize=20, arrowstyle='->',
                              connectionstyle='arc3,rad=0.1')
        
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
                node_colors.append('lightgray')
                node_sizes.append(500)
        
        # Dessiner les nœuds
        nx.draw_networkx_nodes(G_tree, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.9, ax=ax, 
                              edgecolors='black', linewidths=2)
        
        # Mettre en évidence le nœud de départ (racine)
        nx.draw_networkx_nodes(G_tree, pos, nodelist=[start_node], 
                              node_color='green', node_size=900, 
                              alpha=1.0, ax=ax, edgecolors='darkgreen', linewidths=4)
        
        # Dessiner les labels
        labels = {node: str(node) for node in G_tree.nodes()}
        nx.draw_networkx_labels(G_tree, pos, labels, font_size=11, 
                               font_weight='bold', ax=ax)
        
        # Label spécial pour la racine
        start_label = {start_node: f'RACINE\n{start_node}'}
        nx.draw_networkx_labels(G_tree, pos, start_label, font_size=13, 
                               font_weight='bold', font_color='darkgreen', ax=ax)
        
        # Ajouter des lignes de niveau (optionnel)
        for level in range(max_level + 1):
            y_pos = max_level - level
            ax.axhline(y=y_pos, color='gray', linestyle='--', alpha=0.3, linewidth=1)
            ax.text(-max([len(nodes_by_level.get(l, [])) for l in range(max_level + 1)])/2 - 0.5, 
                   y_pos, f'Niveau {level}', fontsize=10, 
                   verticalalignment='center', color='gray')
        
        # Ajouter titre
        title = f'Arbre BFS Hiérarchique - Départ: Station {start_node}\n'
        title += f'Nombre de nœuds: {len(parcours)} | Profondeur: {max_level} niveaux'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        ax.set_aspect('equal')
        
        # Ajouter légende
        from matplotlib.patches import Patch
        
        legend_elements = [
            Patch(facecolor='green', label=f'Racine (Station {start_node})'),
            Patch(facecolor='lightblue', label='Nœuds (gradient = ordre de visite)'),
            Patch(facecolor='blue', edgecolor='blue', label='Arêtes de l\'arbre (parent → enfant)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Ajouter barre de couleur
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                   norm=plt.Normalize(vmin=0, vmax=len(parcours)-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', 
                           fraction=0.02, pad=0.04)
        cbar.set_label('Ordre de visite BFS', rotation=270, labelpad=20)
        
        plt.tight_layout()
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        output_path = standardize_path(os.path.join(results_dir, file_name))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Visualisation arbre BFS sauvegardée dans: {output_path}")
        plt.close()

