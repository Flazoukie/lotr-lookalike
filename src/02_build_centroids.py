# src/02_build_centroids.py
"""
Loads artifacts/gallery_embeddings.npz and computes a centroid (mean embedding)
per character. Saves artifacts/centroids.npz with:
 - centroids: [K, D]
 - classes: [K]
"""

import numpy as np
from pathlib import Path

def l2_normalize(x, eps=1e-10):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)

def main():
    project_root = Path(__file__).resolve().parent.parent
    artifacts = project_root / "artifacts"
    data = np.load(artifacts / "gallery_embeddings.npz", allow_pickle=True)
    E = data["embeddings"]        # [N, D]
    y = data["labels"]            # [N]
    classes = list(data["classes"])
    K = len(classes)
    D = E.shape[1]

    centroids = np.zeros((K, D), dtype=np.float32)
    for k in range(K):
        Ek = E[y == k]
        if Ek.shape[0] == 0:
            raise RuntimeError(f"No embeddings for class {classes[k]}")
        centroids[k] = Ek.mean(axis=0)
    centroids = l2_normalize(centroids)

    np.savez(artifacts / "centroids.npz", centroids=centroids, classes=np.array(classes))
    print(f"Saved {K} centroids to artifacts/centroids.npz")

if __name__ == "__main__":
    main()
