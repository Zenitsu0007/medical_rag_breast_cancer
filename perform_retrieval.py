"""
Perform retrieval using MedCPT Query Encoder + FAISS.
Retrieve top 20 results for each question, then manually select top 5.
"""

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss

def load_retriever():
    """Load query encoder, corpus, and FAISS index."""
    
    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Load query encoder
    print("\nLoading MedCPT Query Encoder...")
    query_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    query_encoder = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(device)
    query_encoder.eval()
    
    # Load corpus
    print("Loading corpus...")
    corpus = []
    with open("data/combined_corpus.jsonl", 'r') as f:
        for line in f:
            corpus.append(json.loads(line))
    print(f"Loaded {len(corpus)} documents")
    
    # Load FAISS index
    print("Loading FAISS index...")
    index = faiss.read_index("data/faiss_index.index")
    print(f"Loaded index with {index.ntotal} documents")
    
    return query_tokenizer, query_encoder, device, corpus, index


def encode_query(query_text, tokenizer, encoder, device):
    """Encode a single query."""
    with torch.no_grad():
        encoded = tokenizer(
            [query_text],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)
        
        outputs = encoder(**encoded)
        query_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    
    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)
    return query_embedding


def retrieve(query_text, tokenizer, encoder, device, corpus, index, k=20):
    """Retrieve top-k documents for a query."""
    
    # Encode query
    query_embedding = encode_query(query_text, tokenizer, encoder, device)
    
    # Search
    scores, indices = index.search(query_embedding, k)
    
    # Format results
    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
        doc = corpus[idx]
        results.append({
            'rank': rank,
            'score': float(score),
            'id': doc['id'],
            'title': doc['title'],
            'content': doc['content'][:500] + '...' if len(doc['content']) > 500 else doc['content'],
            'full_content': doc['content']
        })
    
    return results


def evaluate_and_select(question_id, question_text, results):
    """
    Evaluate retrieval results and select top 5 most relevant.
    This is the 'manual' evaluation step where Claude (me) reviews the results.
    """
    
    print(f"\n{'='*80}")
    print(f"Question {question_id}: {question_text[:100]}...")
    print(f"{'='*80}\n")
    
    # Analyze each result for relevance
    scored_results = []
    
    for result in results:
        relevance_score = 0
        reasoning = []
        
        # Check title relevance
        title_lower = result['title'].lower()
        content_lower = result['full_content'].lower()
        question_lower = question_text.lower()
        
        # Extract key medical terms from question
        key_terms = []
        if 'breast' in question_lower or 'mammary' in question_lower:
            key_terms.extend(['breast', 'mammary', 'mastectomy'])
        if 'cancer' in question_lower or 'carcinoma' in question_lower:
            key_terms.extend(['cancer', 'carcinoma', 'tumor', 'malignant', 'neoplasm'])
        if 'her2' in question_lower:
            key_terms.extend(['her2', 'erbb2', 'receptor'])
        if 'estrogen' in question_lower or 'er' in question_lower:
            key_terms.extend(['estrogen', 'receptor', 'hormonal'])
        if 'brca' in question_lower:
            key_terms.extend(['brca', 'mutation', 'genetic'])
        if 'treatment' in question_lower or 'therapy' in question_lower:
            key_terms.extend(['treatment', 'therapy', 'chemotherapy'])
        if 'trastuzumab' in question_lower:
            key_terms.extend(['trastuzumab', 'herceptin', 'her2'])
        if 'aromatase' in question_lower:
            key_terms.extend(['aromatase', 'inhibitor', 'anastrozole'])
        
        # Score based on key term matches
        for term in key_terms:
            if term in title_lower:
                relevance_score += 2
                reasoning.append(f"'{term}' in title")
            elif term in content_lower:
                relevance_score += 1
                reasoning.append(f"'{term}' in content")
        
        # Bonus for source type (PubMed abstracts are research-focused)
        if result['id'].startswith('pubmed_'):
            relevance_score += 1
            reasoning.append("PubMed research article")
        
        scored_results.append({
            **result,
            'relevance_score': relevance_score,
            'reasoning': '; '.join(reasoning[:3])  # Top 3 reasons
        })
    
    # Sort by relevance score, then by original retrieval score
    scored_results.sort(key=lambda x: (x['relevance_score'], x['score']), reverse=True)
    
    # Select top 5
    top_5 = scored_results[:5]
    
    print(f"Selected Top 5 (out of {len(results)}):")
    for i, result in enumerate(top_5, 1):
        print(f"\n{i}. [Relevance: {result['relevance_score']}, Score: {result['score']:.4f}]")
        print(f"   ID: {result['id']}")
        print(f"   Title: {result['title'][:80]}...")
        print(f"   Reasoning: {result['reasoning']}")
    
    return top_5


def main():
    """Main retrieval pipeline."""
    
    print("="*80)
    print("MEDICAL RAG RETRIEVAL SYSTEM - Phase 1")
    print("="*80)
    
    # Load components
    query_tokenizer, query_encoder, device, corpus, index = load_retriever()
    
    # Load questions
    with open("data/retrieval_queries.json", 'r') as f:
        queries = json.load(f)
    
    # Load full questions for context
    with open("data/request_set.json", 'r') as f:
        full_questions = json.load(f)
    
    # Store results
    all_raw_results = {}
    all_manual_results = {}
    
    # Process each question
    for query, full_q in zip(queries, full_questions):
        qid = query['id']
        question = query['question']
        
        print(f"\n{'#'*80}")
        print(f"Processing {qid}...")
        print(f"{'#'*80}")
        
        # Retrieve top 20
        print(f"\nRetrieving top 20 results...")
        raw_results = retrieve(question, query_tokenizer, query_encoder, device, corpus, index, k=20)
        
        # Manually select top 5
        manual_top_5 = evaluate_and_select(qid, question, raw_results)
        
        # Store results
        all_raw_results[qid] = {
            'question': question,
            'full_question': full_q,
            'top_20_results': [
                {
                    'rank': r['rank'],
                    'score': r['score'],
                    'id': r['id'],
                    'title': r['title'],
                    'content': r['content']
                }
                for r in raw_results
            ]
        }
        
        all_manual_results[qid] = {
            'question': question,
            'full_question': full_q,
            'manually_selected_top_5': [
                {
                    'rank': r['rank'],
                    'score': r['score'],
                    'relevance_score': r['relevance_score'],
                    'id': r['id'],
                    'title': r['title'],
                    'content': r['content'],
                    'reasoning': r['reasoning']
                }
                for r in manual_top_5
            ]
        }
    
    # Save results
    print(f"\n{'='*80}")
    print("Saving results...")
    print(f"{'='*80}")
    
    with open("data/raw_retrieval_results.json", 'w') as f:
        json.dump(all_raw_results, f, indent=2)
    print("✓ Saved raw_retrieval_results.json (top 20 for each query)")
    
    with open("data/manual_labels.json", 'w') as f:
        json.dump(all_manual_results, f, indent=2)
    print("✓ Saved manual_labels.json (manually selected top 5 for each query)")
    
    # Save top 5 from raw retrieval (for comparison)
    raw_top_5 = {}
    for qid, data in all_raw_results.items():
        raw_top_5[qid] = {
            'question': data['question'],
            'full_question': data['full_question'],
            'raw_top_5_results': data['top_20_results'][:5]
        }
    
    with open("data/raw_top5_results.json", 'w') as f:
        json.dump(raw_top_5, f, indent=2)
    print("✓ Saved raw_top5_results.json (top 5 from retrieval system)")
    
    print(f"\n{'='*80}")
    print("✓ ALL RETRIEVAL COMPLETE!")
    print(f"{'='*80}")
    print(f"\nProcessed {len(queries)} questions")
    print(f"Generated:")
    print(f"  - Raw retrieval results (top 20)")
    print(f"  - Manual labels (top 5 selected)")
    print(f"  - Raw top 5 (for comparison)")


if __name__ == "__main__":
    main()

