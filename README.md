---
title: LOTR Lookalike
emoji: 🧙
colorFrom: blue
colorTo: green
sdk: gradio
app_file: src/app.py
pinned: true
---

# LOTR Lookalike

This Hugging Face Space allows you to **upload a photo** of a face and find the closest **Lord of the Rings character lookalike** from our gallery.  

The gallery images are loaded dynamically from the `optimized_gallery` folder, so the repository **does not track large image files**. Please upload them via the Space's **Files and versions** tab if running locally.  

## Features

- Upload a face or character image.
- Get top lookalike characters from LOTR.
- Fast and interactive web app using **Gradio**.

## Project structure

lotr-lookalike/
├─ src/
│ ├─ app.py
│ ├─ 01_embed_gallery.py
│ └─ 02_build_centroids.py
├─ optimized_gallery/ # images must be uploaded manually
├─ requirements.txt
└─ README.md


## How to run locally

1. Clone the repo:  

git clone https://github.com/Flazoukie/lotr-lookalike.git
cd lotr-lookalike

2. Create a virtual environment and install dependencies:

python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

3. Run the app:

python src/app.py