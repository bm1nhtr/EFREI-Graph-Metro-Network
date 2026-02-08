"""
Floyd-Warshall - Plus courts chemins entre toutes les paires de nœuds.

Matrice des distances minimales + analyse (ex. station la plus centrale).
Complexité : O(V³). Espace : O(V²). Gère les poids négatifs (pas de cycle négatif).
"""

import os

import numpy as np

from algorithms.utils import EXPORT_DPI, LAYOUT_FLOYD_GUI, SAVEFIG_PNG_OPTIONS, standardize_path


class FloydWarshall:
    """Algorithme de Floyd-Warshall sur le graphe (matrice des distances)."""

    def __init__(self, graph_data):
        """
        Args:
            graph_data: Tableau numpy des arêtes (N, 3) : source, cible, poids.
        """
        self.graph_data = graph_data
        self.n = self._get_n()
        self.dist = None  # matrice n x n après floyd_warshall()

    def _get_n(self) -> int:
        """Nombre de nœuds (stations 0 à n-1)."""
        nodes = set()
        for edge in self.graph_data:
            nodes.add(int(edge[0]))
            nodes.add(int(edge[1]))
        return max(nodes) + 1 if nodes else 0

    def floyd_warshall(self) -> np.ndarray:
        """
        Plus courts chemins entre toutes les paires. Complexité : O(V³).
        Returns:
            Matrice n×n : dist[i][j] = distance minimale de i à j (inf si pas de chemin).
        """
        n = self.n
        inf = float("inf")
        dist = np.full((n, n), inf)
        np.fill_diagonal(dist, 0)

        for edge in self.graph_data:
            u, v, w = int(edge[0]), int(edge[1]), float(edge[2])
            dist[u, v] = min(dist[u, v], w)
            dist[v, u] = min(dist[v, u], w)  # graphe non orienté

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] != inf and dist[k, j] != inf:
                        dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])

        self.dist = dist
        return dist

    def floyd_warshall_steps(self):
        """Floyd-Warshall étape par étape : yield la matrice après chaque k (pour visualisation web). Même complexité O(V³)."""
        n = self.n
        inf = float("inf")
        dist = np.full((n, n), inf)
        np.fill_diagonal(dist, 0)
        for edge in self.graph_data:
            u, v, w = int(edge[0]), int(edge[1]), float(edge[2])
            dist[u, v] = min(dist[u, v], w)
            dist[v, u] = min(dist[v, u], w)

        def matrix_to_list(d):
            return [
                [int(d[i, j]) if d[i, j] != inf else "inf" for j in range(n)]
                for i in range(n)
            ]

        step_index = 0
        yield {
            "step_index": step_index,
            "description": "Matrice initiale : arêtes directes (poids), reste = inf.",
            "matrix": matrix_to_list(dist),
            "k": -1,
        }
        step_index += 1

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] != inf and dist[k, j] != inf:
                        dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])
            yield {
                "step_index": step_index,
                "description": f"Après k = {k} : chemins passant par la station {k} autorisés.",
                "matrix": matrix_to_list(dist.copy()),
                "k": k,
            }
            step_index += 1

        analysis = self.analyse_centrale(dist)
        st_somme, val_somme = analysis["centrale_somme"]
        st_exc, val_exc = analysis["centrale_excentricite"]
        yield {
            "step_index": step_index,
            "description": f"Fin. Station centrale (somme): {st_somme} ; (excentricité): {st_exc}.",
            "matrix": matrix_to_list(dist),
            "k": n - 1,
            "centrale_somme": st_somme,
            "centrale_excentricite": st_exc,
        }

    def analyse_centrale(self, dist: np.ndarray) -> dict:
        """
        Analyse simple : station la plus "centrale".

        - Centralité par somme : station qui minimise la somme des distances vers toutes les autres.
        - Centralité par excentricité : station qui minimise la distance max vers toute autre.

        Returns:
            dict avec 'somme' (station, valeur), 'excentricite' (station, valeur).
        """
        n = dist.shape[0]
        inf = float("inf")
        somme = np.full(n, inf)
        excentricite = np.full(n, inf)

        for i in range(n):
            row = dist[i]
            valid = row[row != inf]
            if len(valid) == 0:
                continue
            somme[i] = np.sum(row) if np.all(np.isfinite(row)) else inf
            excentricite[i] = np.max(row) if np.all(np.isfinite(row)) else inf

        best_somme_idx = int(np.argmin(somme))
        best_exc_idx = int(np.argmin(excentricite))
        return {
            "centrale_somme": (best_somme_idx, float(somme[best_somme_idx])),
            "centrale_excentricite": (best_exc_idx, float(excentricite[best_exc_idx])),
        }

    def sauvegarder_resultats(
        self,
        dist: np.ndarray,
        file_name: str = "floyd_warshall_result.txt",
    ) -> None:
        """Sauvegarde la matrice des distances et l'analyse dans results/FLOYD_WARSHALL/."""
        results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "FLOYD_WARSHALL")
        os.makedirs(results_dir, exist_ok=True)
        output_path = standardize_path(os.path.join(results_dir, file_name))

        n = dist.shape[0]
        inf = float("inf")
        analysis = self.analyse_centrale(dist)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Floyd-Warshall - Distances entre toutes les paires de stations\n")
            f.write(f"# Matrice {n} x {n} (stations 0 à {n - 1})\n")
            f.write("# Valeur = distance minimale en minutes (inf = pas de chemin)\n\n")

            f.write("# Matrice des distances (ligne = source, colonne = cible)\n")
            f.write("# \t" + "\t".join(str(j) for j in range(n)) + "\n")
            for i in range(n):
                row_str = "\t".join(
                    str(int(dist[i, j])) if dist[i, j] != inf else "inf" for j in range(n)
                )
                f.write(f"{i}\t{row_str}\n")

            f.write("\n# Analyse - Station la plus centrale\n")
            st_somme, val_somme = analysis["centrale_somme"]
            st_exc, val_exc = analysis["centrale_excentricite"]
            f.write(
                f"# - Par somme des distances (minimiser la somme vers toutes les autres) : "
                f"Station {st_somme} (somme = {int(val_somme)} min)\n"
            )
            f.write(
                f"# - Par excentricité (minimiser la distance max vers une autre) : "
                f"Station {st_exc} (excentricité = {int(val_exc)} min)\n"
            )

        # Fichier matrice seule (pour lecture facile)
        matrix_path = standardize_path(os.path.join(results_dir, "floyd_warshall_matrix.txt"))
        with open(matrix_path, "w", encoding="utf-8") as f:
            f.write("# Matrice des distances (Floyd-Warshall)\n")
            for i in range(n):
                row_str = "\t".join(
                    str(int(dist[i, j])) if dist[i, j] != inf else "inf" for j in range(n)
                )
                f.write(f"{i}\t{row_str}\n")

        print(f"[OK] Résultats Floyd-Warshall sauvegardés dans: {output_path}")
        print(f"     Station la plus centrale (somme): {st_somme} | (excentricité): {st_exc}")

    def visualiser_matrice(
        self,
        dist: np.ndarray,
        file_name: str = "floyd_warshall_visualization.png",
        fig=None,
    ):
        """
        Visualise la matrice des distances en heatmap.

        Si fig fourni (GUI), dessine dessus. Sinon sauvegarde dans results/FLOYD_WARSHALL/.
        """
        import matplotlib.pyplot as plt

        n = dist.shape[0]
        inf = float("inf")
        plot_data = np.where(np.isfinite(dist) & (dist != inf), dist, np.nan)
        # Pour la colorbar, utiliser une valeur max finie
        vmax = np.nanmax(plot_data) if np.any(np.isfinite(plot_data)) else 1

        interactive = fig is not None
        if not interactive:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig.clear()
            ax = fig.add_subplot(111)

        im = ax.imshow(plot_data, cmap="YlOrRd", aspect="equal", vmin=0, vmax=vmax)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(range(n))
        ax.set_yticklabels(range(n))
        ax.set_xlabel("Station (arrivée)")
        ax.set_ylabel("Station (départ)")
        analysis = self.analyse_centrale(dist)
        st_somme, _ = analysis["centrale_somme"]
        st_exc, _ = analysis["centrale_excentricite"]
        ax.set_title(
            "Floyd-Warshall : matrice des distances (min)\n"
            f"Station centrale (somme): {st_somme} | (excentricité): {st_exc}"
        )
        plt.colorbar(im, ax=ax, label="Distance (min)")

        if interactive:
            fig.subplots_adjust(**LAYOUT_FLOYD_GUI)
        if not interactive:
            results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "FLOYD_WARSHALL")
            os.makedirs(results_dir, exist_ok=True)
            out_path = standardize_path(os.path.join(results_dir, file_name))
            plt.savefig(out_path, dpi=EXPORT_DPI, bbox_inches="tight", **SAVEFIG_PNG_OPTIONS)
            plt.close()
            print(f"[OK] Visualisation Floyd-Warshall sauvegardée dans: {out_path}")
        return fig
