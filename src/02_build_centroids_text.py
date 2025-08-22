# src/02_build_centroids_text.py
"""
Text-only centroids script that loads from CSV and saves as CSV.
"""

import numpy as np
import json
import csv
from pathlib import Path


def l2_normalize(x, eps=1e-10):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def main():
    project_root = Path(__file__).resolve().parent.parent
    artifacts = project_root / "artifacts"

    # Load metadata
    with open(artifacts / "embeddings_metadata.json", "r") as f:
        metadata = json.load(f)

    classes = metadata["classes"]
    embedding_dim = metadata["embedding_dim"]
    num_samples = metadata["num_samples"]

    print(f"Loading {num_samples} embeddings...")

    # Load embeddings from CSV
    embeddings = []
    labels = []
    paths = []

    with open(artifacts / "embeddings.csv", "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        for row in reader:
            path = row[0]
            label = int(row[1])
            embedding = [float(x) for x in row[2:]]  # Convert to float

            paths.append(path)
            labels.append(label)
            embeddings.append(embedding)

    # Convert to numpy arrays
    E = np.array(embeddings, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    K = len(classes)
    D = E.shape[1]

    # Compute centroids
    print("Computing centroids...")
    centroids = np.zeros((K, D), dtype=np.float32)
    for k in range(K):
        Ek = E[y == k]
        if Ek.shape[0] == 0:
            raise RuntimeError(f"No embeddings for class {classes[k]}")
        centroids[k] = Ek.mean(axis=0)
    centroids = l2_normalize(centroids)

    # Save centroids as CSV
    print("Saving centroids as CSV...")
    with open(artifacts / "centroids.csv", "w", newline="") as f:
        writer = csv.writer(f)
        # Header: class_name, centroid_0, centroid_1, ..., centroid_D-1
        header = ["class_name"] + [f"centroid_{i}" for i in range(D)]
        writer.writerow(header)

        # Data rows
        for k in range(K):
            row = [classes[k]] + [f"{centroids[k, j]:.6f}" for j in range(D)]
            writer.writerow(row)

    # Save centroids metadata
    centroids_metadata = {
        "num_classes": int(K),
        "embedding_dim": int(D),
        "classes": classes,
        "normalization": "L2 normalized",
        "file": "centroids.csv - CSV with columns: class_name,centroid_0,centroid_1,..."
    }

    with open(artifacts / "centroids_metadata.json", "w") as f:
        json.dump(centroids_metadata, f, indent=2)

    print(f"Saved {K} centroids:")
    print(f"  - centroids.csv: {centroids.shape} centroids as text")
    print(f"  - centroids_metadata.json: metadata")


if __name__ == "__main__":
    main()