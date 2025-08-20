# src/app.py
"""
Simple Gradio app:
 - Embeds the uploaded image using the same ResNet backbone
 - Computes cosine similarity to each class centroid (dot product)
 - Returns top-3 matches and the closest gallery example image
"""

from pathlib import Path
import numpy as np
from PIL import Image
import torch, torch.nn as nn
import torchvision as tv
import gradio as gr

from pathlib import Path

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

G = np.load(artifacts / "gallery_embeddings.npz", allow_pickle=True)
E = G["embeddings"]            # [N, D], L2-normalized
paths = [str(p) for p in G["paths"]]
labels = G["labels"]
classes = [str(c) for c in G["classes"]]

C = np.load(artifacts / "centroids.npz", allow_pickle=True)
centroids = C["centroids"]     # [K, D], L2-normalized

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
    emb = to_embedding(img)    # [D]
    sims = centroids @ emb     # [K]
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
