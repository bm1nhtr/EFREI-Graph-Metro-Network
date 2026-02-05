#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projet Graph - Réseau Métro - Interface graphique desktop (Tkinter)
Affichage dynamique du graphe, menu algorithmes, visualisation en couleur.
Lancer depuis la racine du projet : python interface/gui.py
"""

import sys
import os

# Racine du projet (parent du dossier interface/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import tkinter as tk
from tkinter import ttk, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from algorithms.utils import load_graph_data
from algorithms.visualize_metro import visualize_metro_network
from algorithms.bfs import BFS
from algorithms.prim import Prim
from algorithms.bellman_ford import BellmanFord
from algorithms.dfs import DFS
from algorithms.dijkstra import Dijkstra
from algorithms.kruskal import Kruskal

# 19 stations (0-18)
N_STATIONS = 19
DEFAULT_START = 10


class MetroGraphApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Projet Graph - Réseau Métro (19 stations)")
        self.root.minsize(900, 700)
        self.root.geometry("1200x800")

        try:
            self.graph_data = load_graph_data("metro_network.npy")
        except FileNotFoundError:
            messagebox.showerror(
                "Erreur",
                "Fichier data/metro_network.npy introuvable.\nExécutez d'abord : python algorithms/graph_network.py",
            )
            sys.exit(1)

        self.fig = None
        self.canvas = None
        self.toolbar = None
        self._build_ui()

    def _build_ui(self):
        # Panneau gauche : menu
        left = ttk.Frame(self.root, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="Réseau Métro", font=("", 14, "bold")).pack(pady=(0, 10))
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        ttk.Label(left, text="Station de départ (0–18):").pack(anchor=tk.W)
        self.spin_start = ttk.Spinbox(left, from_=0, to=N_STATIONS - 1, width=6)
        self.spin_start.set(DEFAULT_START)
        self.spin_start.pack(anchor=tk.W, pady=(0, 15))

        ttk.Label(left, text="Visualisation:", font=("", 11, "bold")).pack(anchor=tk.W, pady=(5, 5))
        ttk.Button(left, text="Graphe métro", command=self.show_metro).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="BFS (parcours)", command=self.show_bfs).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="BFS (arbre)", command=self.show_bfs_tree).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Prim (MST)", command=self.show_prim).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Bellman-Ford (PCC)", command=self.show_bellman_ford).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="DFS (parcours)", command=self.show_dfs).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="DFS (arbre)", command=self.show_dfs_tree).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Dijkstra (PCC)", command=self.show_dijkstra).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Kruskal (MST)", command=self.show_kruskal).pack(fill=tk.X, pady=2)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        ttk.Button(left, text="Quitter", command=self.root.quit).pack(fill=tk.X, pady=5)

        # Zone matplotlib à droite
        right = ttk.Frame(self.root, padding=5)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(10, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, right)
        self.toolbar.update()

        # Afficher le graphe métro au démarrage
        self.root.after(100, self.show_metro)

    def _get_start_node(self):
        try:
            v = int(self.spin_start.get())
            if 0 <= v < N_STATIONS:
                return v
        except ValueError:
            pass
        return DEFAULT_START

    def _draw_on_fig(self, draw_func, *args, **kwargs):
        """Utilise fig pour dessiner (mode GUI) puis rafraîchit le canvas."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        kwargs["fig"] = self.fig
        if "ax" in draw_func.__code__.co_varnames:
            kwargs["ax"] = ax
        draw_func(*args, **kwargs)
        self.canvas.draw()

    def show_metro(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        visualize_metro_network(fig=self.fig, ax=ax)
        self.canvas.draw()

    def show_bfs(self):
        start = self._get_start_node()
        bfs = BFS(self.graph_data)
        parcours = bfs.parcourir_bfs(start_node=start)
        bfs.visualiser_parcours(parcours, start_node=start, fig=self.fig)
        self.canvas.draw()

    def show_bfs_tree(self):
        start = self._get_start_node()
        bfs = BFS(self.graph_data)
        parcours = bfs.parcourir_bfs(start_node=start)
        bfs.visualiser_arbre_bfs(parcours, start_node=start, fig=self.fig)
        self.canvas.draw()

    def show_dfs(self):
        """Affiche le parcours DFS sur le graphe."""
        start = self._get_start_node()
        dfs = DFS(self.graph_data)
        parcours, parent = dfs.parcourir_dfs(start_node=start)
        dfs.visualiser_parcours(parcours,start_node=start,fig=self.fig)
        self.canvas.draw()

    def show_dfs_tree(self):
        """Affiche l'arbre DFS sur le graphe."""
        start = self._get_start_node()
        dfs = DFS(self.graph_data)
        parcours, parent = dfs.parcourir_dfs(start_node=start)
        dfs.visualiser_arbre_dfs(parcours, parent, start_node=start, fig=self.fig)
        self.canvas.draw()

    def show_prim(self):
        start = self._get_start_node()
        prim = Prim(self.graph_data)
        mst_edges, total_weight = prim.prim_mst(start_node=start)
        prim.visualiser_mst(mst_edges, total_weight, start_node=start, fig=self.fig)
        self.canvas.draw()

    def show_bellman_ford(self):
        start = self._get_start_node()
        bf = BellmanFord(self.graph_data)
        distances, predecessors, has_neg = bf.bellman_ford(start_node=start)
        bf.visualiser_parcours(distances, predecessors, start, has_neg, fig=self.fig)
        self.canvas.draw()

    def show_dijkstra(self):
        """Affiche les plus courts chemins avec Dijkstra."""
        start = self._get_start_node()
        dijkstra = Dijkstra(self.graph_data)
        distances, predecessors = dijkstra.dijkstra(start_node=start)
        dijkstra.visualiser_parcours(distances,predecessors,start_node=start,fig=self.fig )

        self.canvas.draw()

    def show_kruskal(self):
        """Affiche l'arbre couvrant minimum avec Kruskal."""
        start = self._get_start_node()
        kr = Kruskal(self.graph_data)
        mst_edges, total_weight = kr.kruskal_mst(start_node=start)
        kr.visualiser_mst(mst_edges,total_weight,start_node=start, fig=self.fig)
        self.canvas.draw()

    

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = MetroGraphApp()
    app.run()
