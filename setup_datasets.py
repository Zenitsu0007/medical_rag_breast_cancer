"""
Script to download and prepare datasets for the medical RAG project.
This includes:
1. MedRAG textbooks corpus
2. Breast cancer PubMed abstracts (formatted to match textbooks structure)
"""

from datasets import load_dataset
import pandas as pd
import os

def download_and_prepare_datasets():
    """Download and prepare both datasets."""
    
    print("=" * 80)
    print("Downloading MedRAG Textbooks dataset...")
    print("=" * 80)
    textbooks = load_dataset("MedRAG/textbooks", split="train")
    textbooks_df = textbooks.to_pandas()
    print(f"Textbooks dataset: {len(textbooks_df)} snippets")
    print(f"Columns: {textbooks_df.columns.tolist()}")
    print(f"\nSample row:")
    print(textbooks_df.iloc[0])
    
    # Save textbooks
    os.makedirs("data", exist_ok=True)
    textbooks_df.to_json("data/textbooks_corpus.jsonl", orient="records", lines=True)
    print(f"\n✓ Saved to data/textbooks_corpus.jsonl")
    
    print("\n" + "=" * 80)
    print("Downloading Breast Cancer PubMed Abstracts dataset...")
    print("=" * 80)
    pubmed = load_dataset("Gaborandi/breast_cancer_pubmed_abstracts", split="train")
    pubmed_df = pubmed.to_pandas()
    print(f"PubMed dataset: {len(pubmed_df)} abstracts")
    print(f"Columns: {pubmed_df.columns.tolist()}")
    print(f"\nSample row:")
    print(pubmed_df.iloc[0])
    
    print("\n" + "=" * 80)
    print("Formatting PubMed dataset to match textbooks structure...")
    print("=" * 80)
    
    # Format PubMed to match textbooks structure
    # Original columns: pubmed_id, title, abstract
    # Target columns: id, title, content, contents
    formatted_pubmed = pd.DataFrame({
        'id': 'pubmed_' + pubmed_df['pubmed_id'].astype(str),
        'title': pubmed_df['title'],
        'content': pubmed_df['abstract'],
        'contents': pubmed_df['title'] + ' ' + pubmed_df['abstract']
    })
    
    print(f"Formatted PubMed dataset: {len(formatted_pubmed)} snippets")
    print(f"Columns: {formatted_pubmed.columns.tolist()}")
    print(f"\nSample formatted row:")
    print(formatted_pubmed.iloc[0])
    
    # Save formatted PubMed
    formatted_pubmed.to_json("data/pubmed_breast_cancer_corpus.jsonl", orient="records", lines=True)
    print(f"\n✓ Saved to data/pubmed_breast_cancer_corpus.jsonl")
    
    # Create combined corpus
    print("\n" + "=" * 80)
    print("Creating combined corpus...")
    print("=" * 80)
    combined_df = pd.concat([textbooks_df, formatted_pubmed], ignore_index=True)
    combined_df.to_json("data/combined_corpus.jsonl", orient="records", lines=True)
    print(f"Combined corpus: {len(combined_df)} snippets")
    print(f"  - Textbooks: {len(textbooks_df)}")
    print(f"  - PubMed: {len(formatted_pubmed)}")
    print(f"✓ Saved to data/combined_corpus.jsonl")
    
    return textbooks_df, formatted_pubmed, combined_df

if __name__ == "__main__":
    download_and_prepare_datasets()

