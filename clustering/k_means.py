from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd


def perform_kmeans(data, n_clusters=None):
    if n_clusters is not None:
        # Manual selection of the number of clusters
        kmeans = KMeans(n_clusters=n_clusters)
        cluster_labels = kmeans.fit_predict(data)
        cluster_centers = kmeans.cluster_centers_
        inertia = kmeans.inertia_
    else:
        # Automatic selection using the Silhouette Score
        silhouette_scores = []
        inertias = []
        for k in range(2, 11):  # Try different values of k
            kmeans = KMeans(n_clusters=k)
            cluster_labels = kmeans.fit_predict(data)
            silhouette_avg = silhouette_score(data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            inertias.append(kmeans.inertia_)

        # Determine the optimal number of clusters based on silhouette score
        n_clusters = np.argmax(silhouette_scores) + 2  # Add 2 because range starts from 2
        kmeans = KMeans(n_clusters=n_clusters)
        cluster_labels = kmeans.fit_predict(data)
        cluster_centers = kmeans.cluster_centers_
        inertia = kmeans.inertia_

    cluster_centers_df = pd.DataFrame(cluster_centers,
                                      columns=[f'Cluster Center {i + 1}' for i in range(cluster_centers.shape[1])])
    # Round the values in the DataFrame to 2 decimals
    cluster_centers_rounded = cluster_centers_df.round(2)
    cluster_centers_rounded_list = cluster_centers_rounded.values.tolist()

    inertia = np.round(inertia, decimals=2)
    return cluster_labels, cluster_centers_rounded_list, inertia, n_clusters

