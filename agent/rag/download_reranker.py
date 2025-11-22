#!/usr/bin/env python3
"""
Script to download reranker models locally for offline use.

Usage:
    python -m agent.rag.download_reranker
    python -m agent.rag.download_reranker --model cross-encoder/ms-marco-MiniLM-L-12-v2 --output ./models/reranker
"""

import argparse
import os
import sys
from pathlib import Path

def download_reranker_model(model_name: str, output_dir: str):
    """
    Download a reranker model locally.
    
    Args:
        model_name: Hugging Face model identifier (e.g., 'cross-encoder/ms-marco-MiniLM-L-12-v2')
        output_dir: Local directory to save the model
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print("ERROR: sentence-transformers not installed. Install it with: pip install sentence-transformers")
        sys.exit(1)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading reranker model '{model_name}' to '{output_dir}'...")
    print("This may take a few minutes on first download...")
    
    try:
        # Method 1: Try using huggingface_hub if available (cleaner approach)
        try:
            from huggingface_hub import snapshot_download
            print("Using huggingface_hub to download model...")
            snapshot_download(
                repo_id=model_name,
                local_dir=str(output_path),
                local_dir_use_symlinks=False
            )
            print(f"✓ Model successfully downloaded to: {output_path.absolute()}")
            print(f"\nUpdate your rag.yaml with:")
            print(f"  reranker_model: \"{output_path.absolute()}\"")
            return
        except ImportError:
            print("huggingface_hub not available, using sentence-transformers method...")
        except Exception as e:
            print(f"huggingface_hub download failed: {e}, trying alternative method...")
        
        # Method 2: Use CrossEncoder to load and save
        print("Loading model with CrossEncoder (this will download if not cached)...")
        model = CrossEncoder(model_name)
        
        # Save the model to the output directory
        print(f"Saving model to: {output_path}")
        model.save(str(output_path))
        
        print(f"✓ Model successfully saved to: {output_path.absolute()}")
        print(f"\nUpdate your rag.yaml with:")
        print(f"  reranker_model: \"{output_path.absolute()}\"")
        
    except Exception as e:
        print(f"ERROR: Failed to download model: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify the model name is correct")
        print("3. Install huggingface_hub for better downloads: pip install huggingface_hub")
        print("4. Some models may require authentication - set HUGGINGFACE_TOKEN environment variable")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download reranker models locally")
    parser.add_argument(
        "--model",
        default="cross-encoder/ms-marco-MiniLM-L-12-v2",
        help="Model identifier (default: cross-encoder/ms-marco-MiniLM-L-12-v2)"
    )
    parser.add_argument(
        "--output",
        default="./models/reranker",
        help="Output directory (default: ./models/reranker)"
    )
    
    args = parser.parse_args()
    download_reranker_model(args.model, args.output)


if __name__ == "__main__":
    main()

