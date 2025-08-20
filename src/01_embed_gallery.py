# src/01_embed_gallery.py
"""
Compute and save a normalized embedding for every image in optimized_gallery/<class>/*

Saves artifacts/gallery_embeddings.npz with:
 - embeddings: float32 array [N, D]
 - labels: int32 array [N]
 - paths: object array of strings [N]
 - classes: object array of class names [K]
"""

from pathlib import Path
import numpy as np
from PIL import Image
import torch, torch.nn as nn
import torchvision as tv
from tqdm import tqdm

def get_backbone(device="cpu"):
    # Loads ResNet50 and returns a backbone that outputs pooled features [B, 2048, 1, 1]
    weights = tv.models.ResNet50_Weights.IMAGENET1K_V2
    model = tv.models.resnet50(weights=weights)
    # Remove final FC layer; keep avgpool so output is [B, 2048, 1, 1]
    backbone = nn.Sequential(*list(model.children())[:-1])
    backbone.eval().to(device)
    preprocess = weights.transforms()  # correct resize/crop/normalize for the pretrained weights
    return backbone, preprocess

def l2_normalize(x, eps=1e-10):
    # x: [N, D]
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)

def iter_images(root: Path):
    # yields (class_index, class_name, path)
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    for ci, cls in enumerate(classes):
        for p in sorted((root/cls).glob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                yield ci, cls, p

def main():
    project_root = Path(__file__).resolve().parent.parent
    gallery_root = project_root/ "optimized_gallery"
    artifacts = project_root / "artifacts"
    artifacts.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    backbone, preprocess = get_backbone(device)

    classes = sorted([d.name for d in gallery_root.iterdir() if d.is_dir()])
    if len(classes) == 0:
        raise RuntimeError("No class folders found in optimized_gallery/. Please add optimized_gallery/<class>/* images.")

    paths, labels, embs = [], [], []
    with torch.no_grad():
        for ci, cls, p in tqdm(list(iter_images(gallery_root)), desc="Embedding images"):
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                print("Skipping", p, ":", e)
                continue
            x = preprocess(img).unsqueeze(0).to(device)       # [1, 3, H, W]
            feat = backbone(x).squeeze()                      # [2048] (since avgpool present)
            emb = feat.cpu().numpy().reshape(1, -1)           # [1, 2048]
            embs.append(emb)
            labels.append(ci)
            paths.append(str(p))

    if len(embs) == 0:
        raise RuntimeError("No images were embedded. Check optimized_gallery/ and image formats.")

    embs = np.vstack(embs).astype(np.float32)              # [N, D]
    embs = l2_normalize(embs)                              # normalize rows for cosine similarity
    labels = np.array(labels, dtype=np.int32)
    np.savez(artifacts / "gallery_embeddings.npz",
             embeddings=embs, labels=labels, paths=np.array(paths), classes=np.array(classes))

    print(f"Saved {embs.shape[0]} embeddings for {len(classes)} classes to artifacts/gallery_embeddings.npz")

if __name__ == "__main__":
    main()
