In this section, we include code to calculate evaluation metrics.

## Results

| Metric | Result |
|:--|--:|
| Noise points (`-1`) | 1200 |
| Size of clean clusters | [48, 380, 231, 49, 200, 109, 491] |
| Number of unique clusters | 7 |

### Cluster Distribution

| Cluster | Items | Percentage |
|:--|--:|--:|
| Noise (`-1`) | 1200 | 44.3% |
| Cluster 0 | 48 | 1.8% |
| Cluster 1 | 380 | 14.0% |
| Cluster 2 | 231 | 8.5% |
| Cluster 3 | 49 | 1.8% |
| Cluster 4 | 200 | 7.4% |
| Cluster 5 | 109 | 4.0% |
| Cluster 6 | 491 | 18.1% |

### Representation Scores

| Metric | Score |
|:--|--:|
| Silhouette Score (1024-D Original - Cosine) | 0.052 |
| Silhouette Score (3-D UMAP - Cosine) | 0.498 |
| Davies-Bouldin Index (Lower is better) | 0.620 |

---

## Analysis

Despite the high rejection rate, the presence of heavy noise (`-1`) means that some data points cannot be assigned to any cluster. However, we still obtain seven unique clusters, matching the Cora dataset.

Clusters 0 and 3 are barely above the `min_cluster_size` threshold of 44. Meanwhile, clusters 1, 2, 4, and 6 contain the majority of the clustered data, with cluster 6 being the largest.

The imbalance in cluster assignment and high noise count are primarily caused by an overly aggressive density filter.

### Silhouette Score

The Silhouette Score evaluates clustering quality by comparing the distance between data points within the same cluster to the distance between points across neighboring clusters.

The score ranges from -1 to +1:
* **+1** indicates well-separated, dense clusters.
* **0** indicates overlapping or ambiguous boundaries.
* **-1** indicates incorrect cluster assignments.

The 3-D UMAP score of 0.498 confirms that data points in the same cluster are substantially closer to one another than to points in neighboring clusters.

The 1024-D score of 0.052 reflects the diffuse nature of raw high-dimensional space, where distances between points inside a cluster and outside it are nearly indistinguishable due to high dimensionality and non-linearity. That is why dimensionality reduction is necessary: once UMAP projects the manifold into 3-D, the true density emerges and the Silhouette Score jumps to 0.498.

---

## Usage

These measurements provide a valuable diagnostic baseline for our neural network pipeline.

### Post-Training

After the model is trained, if we substitute the newly learned embeddings back into the clustering pipeline, the noise count should drop and the Silhouette Score should increase. If the noise does not decrease or the score fails to improve, there is likely an architectural issue with the model, such as improper layer selection or training dynamics.

### Pre-Training

For pre-training, we use the seven clean clusters (0–6) alongside a `train_mask` to filter out the noise points. This allows the GNN to learn message passing exclusively from clean, high-confidence cluster structures.