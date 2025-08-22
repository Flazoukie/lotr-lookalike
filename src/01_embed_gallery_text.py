# src/01_embed_gallery_text.py
"""
Text-only version that saves everything as CSV and JSON - guaranteed to pass any scanner.
"""

from pathlib import Path
import numpy as np
import json
import csv
from PIL import Image
import torch, torch.nn as nn
import torchvision as tv
from tqdm import tqdm


def get_backbone(device="cpu"):
    weights = tv.models.ResNet50_Weights.IMAGENET1K_V2
    model = tv.models.resnet50(weights=weights)
    backbone = nn.Sequential(*list(model.children())[:-1])
    backbone.eval().to(device)
    preprocess = weights.transforms()
    return backbone, preprocess


def l2_normalize(x, eps=1e-10):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def embed_image(img_path, backbone, preprocess, device):
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = backbone(x).squeeze().cpu().numpy().reshape(1, -1)
    return l2_normalize(feat)[0]


def main():
    project_root = Path(__file__).resolve().parent.parent
    gallery = project_root / "optimized_gallery"
    artifacts = project_root / "artifacts"
    artifacts.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    backbone, preprocess = get_backbone(device)

    embeddings, labels, paths, classes = [], [], [], []
    class_to_idx = {}

    for class_idx, class_dir in enumerate(sorted(gallery.iterdir())):
        if not class_dir.is_dir():
            continue
        class_to_idx[class_dir.name] = class_idx
        classes.append(class_dir.name)

        for img_path in tqdm(list(class_dir.glob("*.jpg")), desc=f"Embedding {class_dir.name}"):
            emb = embed_image(img_path, backbone, preprocess, device)
            embeddings.append(emb)
            labels.append(class_idx)
            paths.append(str(img_path))

    # Convert to arrays
    E = np.vstack(embeddings).astype(np.float32)
    y = np.array(labels, dtype=np.int32)

    # Save embeddings as CSV (most compatible format possible)
    print("Saving embeddings as CSV...")
    with open(artifacts / "embeddings.csv", "w", newline="") as f:
        writer = csv.writer(f)
        # Header: path, label, embedding_0, embedding_1, ..., embedding_D-1
        header = ["path", "label"] + [f"emb_{i}" for i in range(E.shape[1])]
        writer.writerow(header)

        # Data rows
        for i in range(len(embeddings)):
            row = [paths[i], int(labels[i])] + [f"{E[i, j]:.6f}" for j in range(E.shape[1])]
            writer.writerow(row)

    # Save metadata as JSON
    metadata = {
        "num_samples": int(len(embeddings)),
        "num_classes": int(len(classes)),
        "embedding_dim": int(E.shape[1]),
        "model": "ResNet50_IMAGENET1K_V2",
        "normalization": "L2 normalized",
        "classes": [str(cls) for cls in classes],
        "class_to_idx": {str(k): int(v) for k, v in class_to_idx.items()},
        "files": {
            "embeddings": "embeddings.csv - CSV with columns: path,label,emb_0,emb_1,...",
            "metadata": "embeddings_metadata.json - this file"
        }
    }

    with open(artifacts / "embeddings_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {len(embeddings)} embeddings from {len(classes)} classes:")
    print(f"  - embeddings.csv: {E.shape} embeddings as text")
    print(f"  - embeddings_metadata.json: metadata")


if __name__ == "__main__":
    main()
