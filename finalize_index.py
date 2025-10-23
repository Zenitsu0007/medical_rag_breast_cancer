"""
Finalize the FAISS index from the checkpoint embeddings.
The encoding is complete, now we just need to build and save the index.
"""

import numpy as np
import faiss
import os

def finalize_index():
    """Build FAISS index from checkpoint embeddings."""
    
    checkpoint_path = "data/checkpoint_embeddings.npy"
    final_embeddings_path = "data/faiss_index_embeddings.npy"
    index_path = "data/faiss_index.index"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        return False
    
    print("Loading checkpoint embeddings...")
    embeddings = np.load(checkpoint_path)
    print(f"✓ Loaded embeddings: {embeddings.shape}")
    
    # Save as final embeddings
    print(f"\nSaving final embeddings to {final_embeddings_path}...")
    np.save(final_embeddings_path, embeddings)
    print(f"✓ Saved")
    
    # Normalize for cosine similarity
    print("\nNormalizing embeddings for cosine similarity...")
    faiss.normalize_L2(embeddings)
    print("✓ Normalized")
    
    # Build FAISS index
    print("\nBuilding FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product after normalization = cosine similarity
    index.add(embeddings)
    print(f"✓ Built index with {index.ntotal} documents")
    
    # Save index
    print(f"\nSaving FAISS index to {index_path}...")
    faiss.write_index(index, index_path)
    print(f"✓ Saved")
    
    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"\n✓ Removed checkpoint file")
    
    print("\n" + "=" * 80)
    print("✓ FAISS INDEX READY!")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    finalize_index()

