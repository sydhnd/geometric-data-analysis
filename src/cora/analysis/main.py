import hdbscan
import numpy as np
import umap
from torch_geometric.datasets import Planetoid
from collections import Counter
from sklearn.metrics import davies_bouldin_score, silhouette_score

# core data set
data = Planetoid(root="/tmp/Cora", name="Cora")[0]
embedded_array = np.array(data.x)

# Reduce the number of features from 1,433 to a lower dimension.
# From 2,708 x 1,433 to 2,708 x n_components (features).
# Step 1: Fine-tune n_neighbors and n_components.
reducer = umap.UMAP(
n_neighbors=50,
min_dist=0.0,
n_components=3,
metric="cosine",
random_state=42,)

reduced_embeddings = reducer.fit_transform(embedded_array)


#Step2: Fine-tune min_cluster_size and min_samples.
cluster = hdbscan.HDBSCAN(
    min_cluster_size=43,
    min_samples=35,
    metric="euclidean",
)   

cluster_labels = cluster.fit_predict(reduced_embeddings)

# remove the noise and remove duplicate
unique_clusters = set(cluster_labels)
if -1 in unique_clusters:
    counts = Counter(cluster_labels) 
    print(f"Noise points (-1): {counts.get(-1, 0)}")
    unique_clusters.remove(-1)
    print(f"size of clean cluster: {[counts[cluster] for cluster in unique_clusters]}")  

nos_clusters = len(unique_clusters)
print(f"nos of unique clusters {nos_clusters}")

# additional analysisx_array
# cluster distribution

unique_labels, counts = np.unique(cluster_labels, return_counts=True)
total_points = len(cluster_labels)
print("\n--- Cluster Distribution ---")
for lbl, cnt in zip(unique_labels, counts):
    tag = "Noise (-1)" if lbl == -1 else f"Cluster {lbl}"
    print(f"{tag:>12}: {cnt} items ({cnt / total_points * 100:.1f}%)")
  
# Quantitative Clustering Evaluation
valid_mask = cluster_labels != -1
valid_labels = cluster_labels[valid_mask]
n_valid_clusters = len(set(valid_labels))

print("\n--- Representation Scores ---")
if n_valid_clusters >= 2:
    score_orig = silhouette_score(
        embedded_array[valid_mask],
        valid_labels,
        metric="cosine",
    )
    score_umap = silhouette_score(
        reduced_embeddings[valid_mask],
        valid_labels,
        metric="euclidean",
    )
    db_score = davies_bouldin_score(
        reduced_embeddings[valid_mask],
        valid_labels,
    )

    print(f"Silhouette Score (1024-D Original - Cosine): {score_orig:.3f}")
    print(f"Silhouette Score (3-D UMAP - cosine):    {score_umap:.3f}")
    print(f"Davies-Bouldin Index (Lower is better):      {db_score:.3f}")
