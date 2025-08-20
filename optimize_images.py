#!/usr/bin/env python3
"""
Image optimization script for Gradio gallery
Compresses and resizes images to be suitable for Hugging Face deployment
Recursively optimizes all images in subfolders and saves them in optimized_gallery
"""

import os
from PIL import Image
import argparse

def optimize_image(input_path, output_path, max_size=(800, 800), quality=85, max_file_size_mb=2):
    try:
        with Image.open(input_path) as img:
            # Convert to RGB if necessary (handles RGBA, etc.)
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')

            # Resize while maintaining aspect ratio
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

            # Save with compression
            img.save(output_path, 'JPEG', quality=quality, optimize=True)

            # Check file size and reduce quality if needed
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            if file_size_mb > max_file_size_mb:
                quality = max(30, quality - 20)
                img.save(output_path, 'JPEG', quality=quality, optimize=True)

            print(f"✓ Optimized: {input_path} -> {output_path} ({file_size_mb:.2f} MB, Quality: {quality})")
    except Exception as e:
        print(f"✗ Error processing {input_path}: {e}")

def optimize_gallery(input_dir, output_dir="optimized_gallery", **kwargs):
    """Recursively optimize images in all subfolders of the gallery"""
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    for root, dirs, files in os.walk(input_dir):
        # Compute relative path to preserve subfolder structure
        rel_path = os.path.relpath(root, input_dir)
        out_folder = os.path.join(output_dir, rel_path)
        os.makedirs(out_folder, exist_ok=True)

        for filename in files:
            if filename.lower().endswith(supported_formats):
                input_path = os.path.join(root, filename)
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(out_folder, f"{base_name}.jpg")
                optimize_image(input_path, output_path, **kwargs)

def main():
    parser = argparse.ArgumentParser(description="Optimize Gradio gallery images recursively")
    parser.add_argument("input_dir", help="Input gallery directory")
    parser.add_argument("--output_dir", default="optimized_gallery", help="Output directory")
    parser.add_argument("--max_width", type=int, default=800, help="Maximum width")
    parser.add_argument("--max_height", type=int, default=800, help="Maximum height")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality (1-100)")
    parser.add_argument("--max_size_mb", type=float, default=2.0, help="Maximum file size in MB")

    args = parser.parse_args()

    optimize_gallery(
        args.input_dir,
        output_dir=args.output_dir,
        max_size=(args.max_width, args.max_height),
        quality=args.quality,
        max_file_size_mb=args.max_size_mb
    )

if __name__ == "__main__":
    main()
