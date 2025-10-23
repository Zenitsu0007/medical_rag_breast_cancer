"""
Safe version of retriever building with manual checkpointing.
This version processes in smaller chunks and saves progress more frequently.
"""

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import faiss
import os

def load_corpus(corpus_path):
    """Load corpus from JSONL."""
    corpus = []
    with open(corpus_path, 'r') as f:
        for line in f:
            corpus.append(json.loads(line))
    return corpus

def encode_documents_safe(article_tokenizer, article_encoder, texts, device, 
                          batch_size=16, checkpoint_path="data/checkpoint_embeddings.npy",
                          start_idx=0):
    """Encode documents with frequent checkpointing and error handling."""
    print(f"\nEncoding {len(texts)} documents (starting from index {start_idx})...")
    
    embeddings_list = []
    
    # Load existing checkpoint if resuming
    if start_idx > 0 and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint...")
        existing_embeddings = np.load(checkpoint_path)
        print(f"Loaded {existing_embeddings.shape[0]} existing embeddings")
        embeddings_list.append(existing_embeddings)
    
    with torch.no_grad():
        for i in tqdm(range(start_idx, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            
            # Clean texts - handle None and non-string values
            cleaned_texts = []
            for t in batch_texts:
                if t is None:
                    cleaned_texts.append("")
                elif not isinstance(t, str):
                    cleaned_texts.append(str(t))
                else:
                    cleaned_texts.append(t)
            
            try:
                # Tokenize
                encoded = article_tokenizer(
                    cleaned_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(device)
                
                # Get embeddings
                outputs = article_encoder(**encoded)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings_list.append(batch_embeddings.cpu().numpy())
                
                # Save checkpoint every 50 batches (~800 documents with batch_size=16)
                if len(embeddings_list) > 1 and (len(embeddings_list) - 1) % 50 == 0:
                    checkpoint_emb = np.vstack(embeddings_list)
                    np.save(checkpoint_path, checkpoint_emb)
                    current_count = checkpoint_emb.shape[0]
                    print(f"\n✓ Checkpoint: {current_count}/{len(texts)} docs ({100*current_count/len(texts):.1f}%)")
                    
            except Exception as e:
                print(f"\n⚠ Error at index {i}: {e}")
                print(f"Batch size: {len(cleaned_texts)}, First text: {cleaned_texts[0][:100] if cleaned_texts else 'empty'}")
                # Create zero embeddings for this batch
                zero_emb = np.zeros((len(cleaned_texts), 768))  # MedCPT output dimension
                embeddings_list.append(zero_emb)
                continue
    
    # Final save
    final_embeddings = np.vstack(embeddings_list)
    np.save(checkpoint_path, final_embeddings)
    print(f"\n✓ Final checkpoint: {final_embeddings.shape[0]} documents")
    
    return final_embeddings

def main():
    """Build retriever with safe checkpointing."""
    
    # Setup device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Load corpus
    print("\nLoading corpus...")
    corpus_path = "data/combined_corpus.jsonl"
    corpus = load_corpus(corpus_path)
    print(f"Loaded {len(corpus)} documents")
    
    # Extract texts
    doc_texts = [doc['contents'] for doc in corpus]
    
    # Paths
    checkpoint_path = "data/checkpoint_embeddings.npy"
    final_path = "data/faiss_index_embeddings.npy"
    
    # Check for existing work
    start_idx = 0
    if os.path.exists(checkpoint_path):
        checkpoint_emb = np.load(checkpoint_path)
        start_idx = checkpoint_emb.shape[0]
        print(f"\n🎉 Found checkpoint with {start_idx} documents!")
        print(f"Progress: {100*start_idx/len(corpus):.1f}%")
        
        if start_idx >= len(corpus):
            print("✓ All documents already encoded!")
            embeddings = checkpoint_emb
        else:
            print(f"Need to encode {len(corpus) - start_idx} more documents")
            
            # Load models
            print("\nLoading MedCPT models...")
            article_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
            article_encoder = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(device)
            article_encoder.eval()
            
            # Continue encoding
            embeddings = encode_documents_safe(
                article_tokenizer, article_encoder, doc_texts, device,
                batch_size=16, checkpoint_path=checkpoint_path, start_idx=start_idx
            )
    else:
        print("\nNo checkpoint found. Starting from beginning...")
        
        # Load models
        print("Loading MedCPT models...")
        article_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
        article_encoder = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(device)
        article_encoder.eval()
        
        # Encode all documents
        embeddings = encode_documents_safe(
            article_tokenizer, article_encoder, doc_texts, device,
            batch_size=16, checkpoint_path=checkpoint_path
        )
    
    # Save final embeddings
    print(f"\nSaving final embeddings to {final_path}...")
    np.save(final_path, embeddings)
    print(f"✓ Saved {embeddings.shape[0]} embeddings")
    
    # Build FAISS index
    print("\nBuilding FAISS index...")
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Save index
    index_path = "data/faiss_index.index"
    faiss.write_index(index, index_path)
    print(f"✓ Saved FAISS index with {index.ntotal} documents to {index_path}")
    
    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("✓ Removed checkpoint file")
    
    print("\n" + "=" * 80)
    print("✓ ALL DONE!")
    print("=" * 80)

if __name__ == "__main__":
    main()

