#!/usr/bin/env python3
"""
Projet Graph - Réseau Métro - Application Web
Permet d'accéder au projet depuis un navigateur (réseau local ou déploiement).
Lancer depuis la racine du projet : python interface/app.py
"""

import io
import os
import sys

# Racine du projet (parent du dossier interface/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Backend matplotlib sans interface graphique (pour serveur)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file
from matplotlib.figure import Figure

from algorithms.bellman_ford import BellmanFord
from algorithms.bfs import BFS
from algorithms.dfs import DFS
from algorithms.dijkstra import Dijkstra
from algorithms.kruskal import Kruskal
from algorithms.prim import Prim
from algorithms.floyd_warshall import FloydWarshall
from algorithms.utils import load_graph_data
from algorithms.visualize_metro import visualize_metro_network

import networkx as nx

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024

N_STATIONS = 19


def get_graph_data():
    """Charge le graphe depuis data/metro_network.npy. Retourne None si absent."""
    try:
        return load_graph_data("metro_network.npy")
    except FileNotFoundError:
        return None


def get_bellman_graph_data():
    """Graphe avec 1 poids négatif (option A). Retourne None si absent."""
    try:
        return load_graph_data("metro_network_bellman.npy")
    except FileNotFoundError:
        return None


def get_reference_edges():
    """Liste des arêtes pour la table de référence (station_a, station_b, poids)."""
    ref_path = os.path.join(project_root, "results", "REFERENCE_GRAPHE_VERIFICATION.txt")
    edges = []
    if os.path.exists(ref_path):
        with open(ref_path, encoding="utf-8") as f:
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
    """Page d'accueil : choix de la visualisation et de la station de depart."""
    return render_template(
        "index.html",
        n_stations=N_STATIONS,
        reference_edges=get_reference_edges(),
        graph_available=get_graph_data() is not None,
        graph_bellman_available=get_bellman_graph_data() is not None,
    )


@app.route("/steps")
def steps_page():
    """Page visualisation étape par étape des algorithmes."""
    return render_template(
        "steps.html",
        n_stations=N_STATIONS,
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


def _graph_to_json(data):
    """Construit le dict nodes/edges à partir des données graphe."""
    nodes = set()
    edges = []
    for row in data:
        u, v, w = int(row[0]), int(row[1]), float(row[2])
        nodes.add(u)
        nodes.add(v)
        edges.append({"from": u, "to": v, "weight": w})
    return {"nodes": sorted(nodes), "edges": edges}


def _graph_positions_spring(data, width=500, height=450, margin=50):
    """
    Calcule les positions des nœuds avec le même layout que metro_network_visualization
    (spring_layout k=2, iterations=50, seed=42). Retourne dict node_id -> [x, y] en pixels.
    """
    G = nx.Graph()
    for row in data:
        u, v, w = int(row[0]), int(row[1]), float(row[2])
        G.add_edge(u, v, weight=w)
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    xs = [pos[n][0] for n in pos]
    ys = [pos[n][1] for n in pos]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    range_x = max_x - min_x or 1
    range_y = max_y - min_y or 1
    out = {}
    for n in pos:
        x = (pos[n][0] - min_x) / range_x * (width - 2 * margin) + margin
        y = (pos[n][1] - min_y) / range_y * (height - 2 * margin) + margin
        out[str(n)] = [round(x, 2), round(y, 2)]
    return out


@app.route("/api/steps/<algo>")
def api_steps(algo):
    """Retourne les étapes de l'algorithme pour la visualisation pas à pas (JSON)."""
    start = request.args.get("start", type=int, default=10)
    if start is None or start < 0 or start >= N_STATIONS:
        start = 10
    data = get_graph_data()
    if algo == "bellman_ford":
        data_bf = get_bellman_graph_data()
        if data_bf is not None:
            data = data_bf
    if data is None:
        return {"error": "Graphe non trouvé."}, 404

    steps = []
    try:
        if algo == "bfs":
            bfs = BFS(data)
            steps = list(bfs.parcourir_bfs_steps(start_node=start))
        elif algo == "dfs":
            dfs = DFS(data)
            steps = list(dfs.parcourir_dfs_steps(start_node=start))
        elif algo == "dijkstra":
            dj = Dijkstra(data)
            steps = list(dj.dijkstra_steps(start_node=start))
        elif algo == "bellman_ford":
            bf = BellmanFord(data)
            steps = list(bf.bellman_ford_steps(start_node=start))
        elif algo == "prim":
            prim = Prim(data)
            steps = list(prim.prim_mst_steps(start_node=start))
        elif algo == "kruskal":
            kr = Kruskal(data)
            steps = list(kr.kruskal_mst_steps(start_node=start))
        elif algo == "floyd_warshall":
            fw = FloydWarshall(data)
            steps = list(fw.floyd_warshall_steps())
        else:
            return {"error": f"Algorithme inconnu: {algo}"}, 404
    except Exception as e:
        return {"error": str(e)}, 500

    # Sérialiser les steps (clés numériques et types JSON-safe)
    steps_json = []
    for s in steps:
        step = {"step_index": s["step_index"], "description": s["description"]}
        if "visited" in s:
            step["visited"] = s["visited"]
        if "queue" in s:
            step["queue"] = s["queue"]
        if "stack" in s:
            step["stack"] = s["stack"]
        if "current_node" in s:
            step["current_node"] = s["current_node"]
        if "distances" in s:
            step["distances"] = s["distances"]
        if "predecessors" in s:
            pred = s["predecessors"]
            step["predecessors"] = {str(k): (v if isinstance(v, list) else [v] if v is not None else []) for k, v in pred.items()}
        if "mst_edges" in s:
            step["mst_edges"] = s["mst_edges"]
        if "total_weight" in s:
            step["total_weight"] = s["total_weight"]
        if "phase" in s:
            step["phase"] = s["phase"]
        if "has_negative_cycle" in s:
            step["has_negative_cycle"] = s["has_negative_cycle"]
        if "matrix" in s:
            step["matrix"] = s["matrix"]
        if "k" in s:
            step["k"] = s["k"]
        if "centrale_somme" in s:
            step["centrale_somme"] = s["centrale_somme"]
        if "centrale_excentricite" in s:
            step["centrale_excentricite"] = s["centrale_excentricite"]
        steps_json.append(step)

    graph = _graph_to_json(data)
    graph["positions"] = _graph_positions_spring(data)
    return {"graph": graph, "steps": steps_json, "start_node": start, "algo": algo}


def _fig_to_png(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    return buf


@app.route("/api/visualize/metro")
def api_visualize_metro():
    """Image PNG du graphe métro (normal ou Bellman selon query param)."""
    use_bellman = request.args.get("bellman", "").lower() in ("1", "true", "yes")
    if use_bellman and get_bellman_graph_data() is not None:
        fig = Figure(figsize=(12, 9))
        ax = fig.add_subplot(111)
        visualize_metro_network(fig=fig, ax=ax, use_bellman_graph=True)
    else:
        if get_graph_data() is None:
            return "Graphe non trouvé. Exécutez: python algorithms/graph_network.py", 404
        fig = Figure(figsize=(12, 9))
        ax = fig.add_subplot(111)
        visualize_metro_network(fig=fig, ax=ax)
    buf = _fig_to_png(fig)
    plt.close(fig)
    return send_file(buf, mimetype="image/png")


@app.route("/api/visualize/metro_bellman")
def api_visualize_metro_bellman():
    """Image PNG du graphe métro avec poids négatif (option A - Bellman)."""
    if get_bellman_graph_data() is None:
        return "Graphe Bellman non trouvé. Exécutez: python algorithms/graph_network.py", 404
    fig = Figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    visualize_metro_network(fig=fig, ax=ax, use_bellman_graph=True)
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
    data = get_bellman_graph_data()
    if data is None:
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
    dfs.visualiser_parcours(parcours, parent, start_node=start, fig=fig)

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


@app.route("/api/visualize/floyd_warshall")
def api_visualize_floyd_warshall():
    """Matrice des distances (toutes paires) + analyse centrale."""
    data = get_graph_data()
    if data is None:
        return "Graphe non trouvé.", 404
    fw = FloydWarshall(data)
    dist = fw.floyd_warshall()
    fig = Figure(figsize=(12, 9))
    fw.visualiser_matrice(dist, fig=fig)
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
