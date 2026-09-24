# Investigating Cora with UMAP and HDBSCAN

This project investigates parameter tuning for dimensionality reduction and unsupervised clustering on the Cora dataset. Using **UMAP** and **HDBSCAN**, high-dimensional node features are mapped into a low-dimensional manifold to evaluate cluster structure against the 7 ground-truth classes.

---

## Repository Structure

```text
├── images/
│   ├── cora.png
│   ├── graphcompare.png
│   └── Figure_2.png
└── src/
    └── cora/
        ├── umap/
        │   ├── main.py
        │   └── readme.md
        ├── hbscan/
        │   ├── main.py
        │   └── readme.md
        └── analysis/
            ├── main.py
            └── readme.md
```

### Modules & Pipeline

* **[1. UMAP Projection (src/cora/umap/)](src/cora/umap/readme.md)**  
  Tuning `n_neighbors` and `n_components` to project raw node features into a 3D manifold without excessive micro-clustering.

* **[2. HDBSCAN Clustering (src/cora/hbscan/)](src/cora/hbscan/readme.md)**  
  Parameter sweeps across `min_cluster_size` and `min_samples` to balance noise point reduction (`-1`) against cluster counts.

* **[3. Analysis & Evaluation (src/cora/analysis/)](src/cora/analysis/readme.md)**  
  Evaluating clustering quality across representations (1024-D raw vs. 3-D UMAP), computing cluster size distributions, and testing downstream utility for GNN training.

---

## Installation & Setup

### 1. Prerequisites

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

Install the required packages:

```bash
pip install torch torchvision torchaudio
pip install torch_geometric
pip install umap-learn hdbscan scikit-learn matplotlib
```

---

## Execution

Run the scripts in order from the repository root:

```bash
# Step 1: Run UMAP projection experiments
python src/cora/umap/main.py

# Step 2: Run HDBSCAN parameter sweeps
python src/cora/hbscan/main.py

# Step 3: Run cluster distribution and metric analysis
python src/cora/analysis/main.py
```

---

## Analysis

Here are the main takeaways from our tests:

* **Finding the 7 classes comes with a noise trade-off:** When we tune HDBSCAN to match the 7 true Cora classes (`min_cluster_size=44`, `min_samples=35`), it works, but the strict density setting throws out 44.3% of the data as noise (`-1`).
* **Raw space vs. 3D UMAP:** In the raw 1024-D space, the silhouette score is practically zero (0.052) because the points are too spread out to tell clusters apart. Once UMAP pulls the structure down into 3-D, the score jumps to 0.498, showing that the clusters really do exist once projected.
* **Small clusters are fragile:** Clusters 0 and 3 have only 48 and 49 points, which is right on the edge of our threshold of 44. If we push `min_cluster_size` any higher, those two groups disappear into noise completely.
* **How this helps the GNN:** Even with the high noise, we can use the clean clusters (0 to 6) with a `train_mask` to pre-train our GNN without feeding it garbage data. After training, we can run the new embeddings back through this pipeline to see if the noise drops and the model actually learned.
