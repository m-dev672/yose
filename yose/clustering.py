import numpy as np
import html
from collections import defaultdict
from sklearn.cluster import SpectralClustering

class Clustering:
    def __init__(self, n_clusters=50, default_distance=1.0, export=True, output_path="clusters.html"):
        self.n_clusters = n_clusters
        self.default_distance = default_distance
        self.export = export
        self.output_path=output_path

    def build_distance_matrix(self, n_samples, combinations, similarities):
        D = np.full((n_samples, n_samples), self.default_distance)
        np.fill_diagonal(D, 0.0)

        for (i, j), dist in zip(combinations, similarities):
            D[i, j] = dist
            D[j, i] = dist

        return D

    def clustering(self, D, combinations, similarities):
        model = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=0
        )

        beta = 1.0
        S = np.exp(-beta * D / D.std())

        labels = model.fit_predict(S)

        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append(idx)
        
        return clusters.values()

    def export_clusters_to_html(self, sentences, clusters):
        html_lines = [
            "<!DOCTYPE html>",
            "<html lang='ja'>",
            "<head>",
            "<meta charset='UTF-8'>",
            "<title>クラスタリング結果</title>",
            "<style>",
            "body { font-family: sans-serif; }",
            ".cluster { margin-bottom: 2em; padding: 1em; border: 1px solid #ccc; border-radius: 10px; background: #f9f9f9; }",
            ".cluster h2 { margin-top: 0; }",
            ".sentence { margin-left: 1em; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>クラスタリング結果</h1>"
        ]

        for i, cluster in enumerate(clusters):
            html_lines.append(f"<div class='cluster'>")
            html_lines.append(f"<h2>クラスタ {i+1}（{len(cluster)}文）</h2>")
            for idx in sorted(cluster):
                sentence = html.escape(sentences[idx].strip())
                html_lines.append(f"<p class='sentence'>{sentence}</p>")
            html_lines.append("</div>")

        html_lines.extend(["</body>", "</html>"])

        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_lines))

        print(f"クラスタリング結果を {self.output_path} に保存しました。")

    def run(self, sentences, combinations, similarities):
        D = self.build_distance_matrix(len(sentences), combinations, similarities)
        clusters = self.clustering(D, combinations, similarities)
        if self.export == True:
            self.export_clusters_to_html(sentences, clusters)
        return clusters