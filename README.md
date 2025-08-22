
## 🧙‍♀️ LOTR Lookalike App

Find out which Lord of the Rings character you look like!
This project uses image recognition, embeddings, and Gradio to match a photo you upload to the closest LOTR character in our gallery.

🚀 Try it on Hugging Face: [LOTR App](https://huggingface.co/spaces/Flazoukie/lotr-lookalike)

## ⚡ How It Works

**Dataset**

- Collected 20 images per character (Google search).

- Stored in optimized_gallery/character_name/.

**Embeddings**

- We use a pretrained ResNet-50 model to compute embeddings for each image.

- Script: 01_embed_gallery.py

**Centroids**

- For each character, we compute a centroid embedding (average).

- Script: 02_build_centroids.py

**Matching**

- The uploaded image is embedded.

- We find the closest centroid and display both the predicted character and the closest gallery image.

**Gradio App**

- Run app.py to launch the interface locally or deploy on Hugging Face.

## 🖥️ Running Locally

- Clone the repo and install requirements:

git clone https://github.com/yourusername/lotr-lookalike.git
cd lotr-lookalike
pip install -r requirements.txt

- Then run the app:

python src/app.py

- The app will be available at http://127.0.0.1:7860.

## 🚧 Known Issues

- The dataset is small, so predictions may be noisy.
- Sometimes the app may assign you a different gender (fun fact: you might end up as Gimli 😅).
- Security scans required us to switch from .npz to CSV/JSON for embeddings.

## 🔮 Future Improvements

- Expand dataset (more characters, more images).
- Fine-tune embeddings with a face-recognition model.

## 📖 Blog Post

I wrote about the full process here:
👉 [Blog Post](https://flazoukie.github.io/data-blog/posts/LOTR-lookalike.html)

## 🙌 Acknowledgments

Built with PyTorch
, Gradio
, and Hugging Face Spaces
.
