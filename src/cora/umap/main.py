import hdbscan
import numpy as np
import umap
from torch_geometric.datasets import Planetoid

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

# Step 2: Fine-tune min_cluster_size and min_samples.
cluster = hdbscan.HDBSCAN(
    min_cluster_size=10,
    min_samples=3,
    metric="euclidean",
)

# train the model
cluster_labels = cluster.fit_predict(reduced_embeddings)

unique_clusters = set(cluster_labels)
if -1 in unique_clusters:
    unique_clusters.remove(-1)

nos_clusters = len(unique_clusters)
print(f"nos of unique clusters {nos_clusters}")
