# src/app_text.py
"""
Text-only app that loads from CSV format.
"""

from pathlib import Path
import numpy as np
import json
import csv
from PIL import Image
import torch, torch.nn as nn
import torchvision as tv
import gradio as gr


# --- backbone (same as embedding script) ---
def get_backbone(device="cpu"):
    weights = tv.models.ResNet50_Weights.IMAGENET1K_V2
    model = tv.models.resnet50(weights=weights)
    backbone = nn.Sequential(*list(model.children())[:-1])
    backbone.eval().to(device)
    preprocess = weights.transforms()
    return backbone, preprocess


def l2_normalize(x, eps=1e-10):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


# --- load artifacts ---
project_root = Path(__file__).resolve().parent.parent
artifacts = project_root / "artifacts"

print("Loading embeddings from CSV...")

# Load embeddings metadata
with open(artifacts / "embeddings_metadata.json", "r") as f:
    emb_metadata = json.load(f)

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
        embedding = [float(x) for x in row[2:]]

        # Convert absolute path to relative path for cross-platform compatibility
        if 'optimized_gallery' in path:
            # Extract just the part after 'optimized_gallery'
            rel_path = 'optimized_gallery' + path.split('optimized_gallery')[1].replace('\\', '/')
            path = str(project_root / rel_path)

        paths.append(path)
        labels.append(label)
        embeddings.append(embedding)

# Convert to numpy arrays
E = np.array(embeddings, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)
classes = emb_metadata["classes"]

print("Loading centroids from CSV...")

# Load centroids from CSV
centroids = []
centroid_classes = []

with open(artifacts / "centroids.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)  # Skip header

    for row in reader:
        class_name = row[0]
        centroid = [float(x) for x in row[1:]]

        centroid_classes.append(class_name)
        centroids.append(centroid)

centroids = np.array(centroids, dtype=np.float32)

print(f"Loaded {len(embeddings)} embeddings and {len(centroids)} centroids")

device = "cuda" if torch.cuda.is_available() else "cpu"
backbone, preprocess = get_backbone(device)


def to_embedding(img: Image.Image):
    # returns a 1D numpy L2-normalized embedding
    img = img.convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = backbone(x).squeeze().cpu().numpy().reshape(1, -1)
    return l2_normalize(feat)[0]


def predict(img: Image.Image):
    emb = to_embedding(img)  # [D]
    sims = centroids @ emb  # [K]
    idx = np.argsort(-sims)[:3]
    top = {classes[i]: float(sims[i]) for i in idx}

    # --- absolute NN (across all classes) ---
    nn_abs_idx = int(np.argmax(E @ emb))
    nn_abs_path = paths[nn_abs_idx]
    nn_abs_class = classes[int(labels[nn_abs_idx])]
    nn_abs_img = Image.open(nn_abs_path).convert("RGB")

    # --- restricted NN (within top class only) ---
    top_class = idx[0]
    mask = labels == top_class
    masked_scores = (E @ emb)[mask]
    nn_local_idx = int(np.argmax(masked_scores))
    nn_global_idx = np.where(mask)[0][nn_local_idx]
    nn_top_path = paths[nn_global_idx]
    nn_top_class = classes[top_class]
    nn_top_img = Image.open(nn_top_path).convert("RGB")

    verdict = f"Top match: {classes[top_class]} (similarity {sims[top_class]:.3f})"
    return top, nn_abs_img, nn_top_img, verdict


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a face or character image"),
    outputs=[
        gr.Label(num_top_classes=3, label="Top matches (cosine similarity)"),
        gr.Image(label="Absolute closest example (may not belong to the three top characters!)"),
        gr.Image(label="Closest picture from top predicted character"),
        gr.Textbox(label="Verdict")
    ],
    title="Which LOTR character are you?",
    description="Matches your image to LOTR characters using image embeddings."
)

if __name__ == "__main__":
    demo.launch(share=False)