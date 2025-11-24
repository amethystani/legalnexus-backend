"""
Download Jina Embeddings v3 Model

Explicitly downloads all model files from Hugging Face to ensure local availability.
"""

from huggingface_hub import snapshot_download
import os

def download_model():
    model_name = "jinaai/jina-embeddings-v3"
    local_dir = "models/jina-embeddings-v3"
    
    print(f"Downloading {model_name} to {local_dir}...")
    
    # Download all files
    path = snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # Ensure actual files are downloaded
        resume_download=True
    )
    
    print(f"\n✅ Model downloaded successfully to: {path}")
    
    # List files
    print("\nDownloaded files:")
    for root, dirs, files in os.walk(path):
        for file in files:
            print(f"  - {file}")

if __name__ == "__main__":
    download_model()
