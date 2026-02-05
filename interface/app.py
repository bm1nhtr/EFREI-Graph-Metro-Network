#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projet Graph - Réseau Métro - Application Web
Permet d'accéder au projet depuis un navigateur (réseau local ou déploiement).
Lancer depuis la racine du projet : python interface/app.py
"""

import io
import sys
import os

# Racine du projet (parent du dossier interface/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Backend matplotlib sans interface graphique (pour serveur)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, send_file
from matplotlib.figure import Figure

from algorithms.utils import load_graph_data
from algorithms.visualize_metro import visualize_metro_network
from algorithms.bfs import BFS
from algorithms.prim import Prim
from algorithms.bellman_ford import BellmanFord
from algorithms.dijkstra import Dijkstra
from algorithms.dfs import DFS
from algorithms.kruskal import Kruskal

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024

N_STATIONS = 19


def get_graph_data():
    try:
        return load_graph_data("metro_network.npy")
    except FileNotFoundError:
        return None


def get_reference_edges():
    """Liste des arêtes pour la table de référence (station_a, station_b, poids)."""
    ref_path = os.path.join(project_root, "results", "REFERENCE_GRAPHE_VERIFICATION.txt")
    edges = []
    if os.path.exists(ref_path):
        with open(ref_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    edges.append((int(parts[0]), int(parts[1]), int(parts[2])))
    if not edges and get_graph_data() is not None:
        for row in get_graph_data():
            edges.append((int(row[0]), int(row[1]), int(row[2])))
    return edges


@app.route("/")
def index():
    return render_template(
        "index.html",
        n_stations=N_STATIONS,
        reference_edges=get_reference_edges(),
        graph_available=get_graph_data() is not None,
    )


@app.route("/api/graph")
def api_graph():
    """Retourne le graphe en JSON (nœuds et arêtes) pour usage frontend optionnel."""
    data = get_graph_data()
    if data is None:
        return {"error": "Fichier metro_network.npy introuvable"}, 404
    nodes = set()
    edges = []
    for row in data:
        u, v, w = int(row[0]), int(row[1]), float(row[2])
        nodes.add(u)
        nodes.add(v)
        edges.append({"from": u, "to": v, "weight": w})
    return {"nodes": sorted(nodes), "edges": edges}


def _fig_to_png(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    return buf


@app.route("/api/visualize/metro")
def api_visualize_metro():
    """Image PNG du graphe métro."""
    if get_graph_data() is None:
        return "Graphe non trouvé. Exécutez d'abord: python algorithms/graph_network.py", 404
    fig = Figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    visualize_metro_network(fig=fig, ax=ax)
    buf = _fig_to_png(fig)
    plt.close(fig)
    return send_file(buf, mimetype="image/png")


@app.route("/api/visualize/bfs")
def api_visualize_bfs():
    start = request.args.get("start", type=int, default=10)
    if start is None or start < 0 or start >= N_STATIONS:
        start = 10
    data = get_graph_data()
    if data is None:
        return "Graphe non trouvé.", 404
    bfs = BFS(data)
    parcours = bfs.parcourir_bfs(start_node=start)
    fig = Figure(figsize=(12, 9))
    bfs.visualiser_parcours(parcours, start_node=start, fig=fig)
    buf = _fig_to_png(fig)
    plt.close(fig)
    return send_file(buf, mimetype="image/png")


@app.route("/api/visualize/bfs_tree")
def api_visualize_bfs_tree():
    start = request.args.get("start", type=int, default=10)
    if start is None or start < 0 or start >= N_STATIONS:
        start = 10
    data = get_graph_data()
    if data is None:
        return "Graphe non trouvé.", 404
    bfs = BFS(data)
    parcours = bfs.parcourir_bfs(start_node=start)
    fig = Figure(figsize=(14, 10))
    bfs.visualiser_arbre_bfs(parcours, start_node=start, fig=fig)
    buf = _fig_to_png(fig)
    plt.close(fig)
    return send_file(buf, mimetype="image/png")


@app.route("/api/visualize/prim")
def api_visualize_prim():
    start = request.args.get("start", type=int, default=10)
    if start is None or start < 0 or start >= N_STATIONS:
        start = 10
    data = get_graph_data()
    if data is None:
        return "Graphe non trouvé.", 404
    prim = Prim(data)
    mst_edges, total_weight = prim.prim_mst(start_node=start)
    fig = Figure(figsize=(12, 9))
    prim.visualiser_mst(mst_edges, total_weight, start_node=start, fig=fig)
    buf = _fig_to_png(fig)
    plt.close(fig)
    return send_file(buf, mimetype="image/png")


@app.route("/api/visualize/bellman_ford")
def api_visualize_bellman_ford():
    start = request.args.get("start", type=int, default=10)
    if start is None or start < 0 or start >= N_STATIONS:
        start = 10
    data = get_graph_data()
    if data is None:
        return "Graphe non trouvé.", 404
    bf = BellmanFord(data)
    distances, predecessors, has_neg = bf.bellman_ford(start_node=start)
    fig = Figure(figsize=(12, 9))
    bf.visualiser_parcours(distances, predecessors, start, has_neg, fig=fig)
    buf = _fig_to_png(fig)
    plt.close(fig)
    return send_file(buf, mimetype="image/png")

@app.route("/api/visualize/dijkstra")
def api_visualize_dijkstra():
    start = request.args.get("start", type=int, default=10)
    if start is None or start < 0 or start >= N_STATIONS:
        start = 10

    data = get_graph_data()
    if data is None:
        return "Graphe non trouvé.", 404

    dj = Dijkstra(data)
    distances, predecessors = dj.dijkstra(start_node=start)

    fig = Figure(figsize=(12, 9))
    dj.visualiser_parcours(distances, predecessors, start, fig=fig)

    buf = _fig_to_png(fig)
    plt.close(fig)
    return send_file(buf, mimetype="image/png")

@app.route("/api/visualize/dfs")
def api_visualize_dfs():
    start = request.args.get("start", type=int, default=10)
    if start is None or start < 0 or start >= N_STATIONS:
        start = 10

    data = get_graph_data()
    if data is None:
        return "Graphe non trouvé.", 404

    dfs = DFS(data)
    parcours, parent = dfs.parcourir_dfs(start_node=start)

    fig = Figure(figsize=(12, 9))
    dfs.visualiser_parcours(parcours,parent,start_node=start, fig=fig)

    buf = _fig_to_png(fig)
    plt.close(fig)
    return send_file(buf, mimetype="image/png")

@app.route("/api/visualize/dfs_tree")
def api_visualize_dfs_tree():
    start = request.args.get("start", type=int, default=10)
    if start is None or start < 0 or start >= N_STATIONS:
        start = 10

    data = get_graph_data()
    if data is None:
        return "Graphe non trouvé.", 404

    dfs = DFS(data)
    parcours, parent = dfs.parcourir_dfs(start_node=start)

    fig = Figure(figsize=(14, 10))
    dfs.visualiser_arbre_dfs(parcours, parent, start_node=start, fig=fig)

    buf = _fig_to_png(fig)
    plt.close(fig)
    return send_file(buf, mimetype="image/png")


@app.route("/api/visualize/kruskal")
def api_visualize_kruskal():
    start = request.args.get("start", type=int, default=10)
    if start is None or start < 0 or start >= N_STATIONS:
        start = 10

    data = get_graph_data()
    if data is None:
        return "Graphe non trouvé.", 404

    kr = Kruskal(data)
    mst_edges, total_weight = kr.kruskal_mst(start_node=start)

    fig = Figure(figsize=(12, 9))
    kr.visualiser_mst(mst_edges, total_weight, start_node=start, fig=fig)

    buf = _fig_to_png(fig)
    plt.close(fig)
    return send_file(buf, mimetype="image/png")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 60)
    print("Projet Graph - Serveur Web")
    print("=" * 60)
    print(f"Ouvrez dans un navigateur: http://localhost:{port}")
    print(f"Depuis un autre appareil (réseau local): http://<IP_DE_CETTE_MACHINE>:{port}")
    print("Arrêt: Ctrl+C")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False)
