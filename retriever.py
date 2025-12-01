"""
Phase 2: Improved Retrieval using BM25 + MedCPT with RRF-2 Fusion

Based on MedRAG paper findings:
- RRF-2 (BM25 + MedCPT) achieves best performance on medical QA
- Reciprocal Rank Fusion combines lexical (BM25) and semantic (MedCPT) retrieval

Reference: Xiong et al., "Benchmarking Retrieval-Augmented Generation for Medicine", ACL 2024
"""

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import re
import time
from typing import List, Dict, Tuple, Any


class HybridRetriever:
    """
    Hybrid retriever combining BM25 (lexical) and MedCPT (semantic) with RRF fusion.
    """
    
    def __init__(self, corpus_path: str = "data/combined_corpus.jsonl", 
                 faiss_index_path: str = "data/faiss_index.index",
                 rrf_k: int = 60):
        """
        Initialize the hybrid retriever.
        
        Args:
            corpus_path: Path to the corpus JSONL file
            faiss_index_path: Path to the FAISS index
            rrf_k: RRF constant (default 60 as per original RRF paper)
        """
        self.rrf_k = rrf_k
        self.device = self._get_device()
        
        print(f"Using device: {self.device}")
        print("\n" + "="*60)
        print("Initializing Hybrid Retriever (BM25 + MedCPT + RRF-2)")
        print("="*60)
        
        # Load corpus
        self.corpus = self._load_corpus(corpus_path)
        
        # Initialize BM25
        self.bm25 = self._init_bm25()
        
        # Initialize MedCPT
        self.query_tokenizer, self.query_encoder = self._init_medcpt()
        
        # Load FAISS index
        self.faiss_index = self._load_faiss(faiss_index_path)
        
        print("\n✓ Hybrid Retriever ready!")
    
    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _load_corpus(self, path: str) -> List[Dict]:
        print("\nLoading corpus...")
        corpus = []
        with open(path, 'r') as f:
            for line in f:
                corpus.append(json.loads(line))
        print(f"  ✓ Loaded {len(corpus)} documents")
        return corpus
    
    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        if text is None:
            return []
        text = str(text).lower()
        # Keep alphanumeric and split
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _init_bm25(self) -> BM25Okapi:
        """Initialize BM25 index."""
        print("\nBuilding BM25 index...")
        
        # Tokenize all documents
        tokenized_corpus = []
        for doc in tqdm(self.corpus, desc="  Tokenizing"):
            text = doc.get('contents', doc.get('content', ''))
            tokens = self._tokenize_for_bm25(text)
            tokenized_corpus.append(tokens)
        
        # Build BM25
        bm25 = BM25Okapi(tokenized_corpus)
        print(f"  ✓ BM25 index built")
        return bm25
    
    def _init_medcpt(self) -> Tuple[Any, Any]:
        """Initialize MedCPT query encoder."""
        print("\nLoading MedCPT Query Encoder...")
        tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        encoder = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(self.device)
        encoder.eval()
        print(f"  ✓ MedCPT loaded on {self.device}")
        return tokenizer, encoder
    
    def _load_faiss(self, path: str) -> faiss.Index:
        """Load FAISS index."""
        print(f"\nLoading FAISS index from {path}...")
        index = faiss.read_index(path)
        print(f"  ✓ FAISS index loaded ({index.ntotal} vectors)")
        return index
    
    def retrieve_bm25(self, query: str, k: int = 100) -> List[Tuple[int, float]]:
        """
        Retrieve top-k documents using BM25.
        
        Returns: List of (doc_idx, score) tuples
        """
        query_tokens = self._tokenize_for_bm25(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        return [(int(idx), float(scores[idx])) for idx in top_indices]
    
    def retrieve_medcpt(self, query: str, k: int = 100) -> List[Tuple[int, float]]:
        """
        Retrieve top-k documents using MedCPT + FAISS.
        
        Returns: List of (doc_idx, score) tuples
        """
        with torch.no_grad():
            encoded = self.query_tokenizer(
                [query],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            outputs = self.query_encoder(**encoded)
            query_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.faiss_index.search(query_embedding, k)
        
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]
    
    def reciprocal_rank_fusion(self, 
                               rankings: List[List[Tuple[int, float]]], 
                               k: int = 20) -> List[Tuple[int, float]]:
        """
        Perform Reciprocal Rank Fusion on multiple ranking lists.
        
        RRF formula: score(d) = sum over all rankings of 1/(k + rank(d))
        where k is a constant (default 60)
        
        Args:
            rankings: List of ranked lists, each containing (doc_idx, original_score)
            k: Final number of results to return
            
        Returns: Fused ranking as list of (doc_idx, rrf_score)
        """
        rrf_scores = {}
        
        for ranking in rankings:
            for rank, (doc_idx, _) in enumerate(ranking, start=1):
                if doc_idx not in rrf_scores:
                    rrf_scores[doc_idx] = 0.0
                rrf_scores[doc_idx] += 1.0 / (self.rrf_k + rank)
        
        # Sort by RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_docs[:k]
    
    def retrieve(self, query: str, k: int = 20, 
                 retrieval_depth: int = 100,
                 method: str = "rrf2") -> Dict[str, Any]:
        """
        Perform hybrid retrieval.
        
        Args:
            query: Query string
            k: Number of final results
            retrieval_depth: Number of candidates from each retriever before fusion
            method: "rrf2" (BM25+MedCPT), "bm25", "medcpt"
            
        Returns: Dict with results and timing info
        """
        start_time = time.time()
        
        if method == "bm25":
            raw_results = self.retrieve_bm25(query, k)
            retriever_used = "BM25"
        elif method == "medcpt":
            raw_results = self.retrieve_medcpt(query, k)
            retriever_used = "MedCPT"
        elif method == "rrf2":
            # Get candidates from both retrievers
            bm25_results = self.retrieve_bm25(query, retrieval_depth)
            medcpt_results = self.retrieve_medcpt(query, retrieval_depth)
            
            # Fuse with RRF
            raw_results = self.reciprocal_rank_fusion([bm25_results, medcpt_results], k)
            retriever_used = "RRF-2 (BM25 + MedCPT)"
        else:
            raise ValueError(f"Unknown method: {method}")
        
        elapsed = time.time() - start_time
        
        # Format results
        results = []
        for rank, (doc_idx, score) in enumerate(raw_results, start=1):
            doc = self.corpus[doc_idx]
            results.append({
                'rank': rank,
                'score': float(score),
                'id': doc['id'],
                'title': doc['title'],
                'content': doc['content'][:500] + '...' if len(doc['content']) > 500 else doc['content'],
                'full_content': doc['content']
            })
        
        return {
            'results': results,
            'retriever': retriever_used,
            'latency_ms': elapsed * 1000,
            'num_results': len(results)
        }


def run_retrieval(retriever: HybridRetriever, 
                  queries_path: str,
                  requests_path: str,
                  output_path: str,
                  method: str = "rrf2",
                  k: int = 20) -> Dict[str, Any]:
    """
    Run retrieval on all queries and save results.
    """
    # Load queries
    with open(queries_path, 'r') as f:
        queries = json.load(f)
    
    with open(requests_path, 'r') as f:
        full_questions = json.load(f)
    
    all_results = {}
    total_latency = 0
    
    print(f"\n{'='*60}")
    print(f"Running {method.upper()} retrieval on {len(queries)} queries")
    print(f"{'='*60}")
    
    for query, full_q in zip(queries, full_questions):
        qid = query['id']
        question = query['question']
        
        print(f"\n[{qid}] {question[:60]}...")
        
        # Retrieve
        result = retriever.retrieve(question, k=k, method=method)
        
        print(f"  → {result['retriever']}, {result['latency_ms']:.1f}ms, {result['num_results']} results")
        
        # Top 3 preview
        for i, r in enumerate(result['results'][:3], 1):
            print(f"    {i}. [{r['id']}] {r['title'][:50]}... (score: {r['score']:.4f})")
        
        total_latency += result['latency_ms']
        
        # Store
        all_results[qid] = {
            'question': question,
            'full_question': full_q,
            'retriever': result['retriever'],
            'latency_ms': result['latency_ms'],
            'top_20_results': [
                {
                    'rank': r['rank'],
                    'score': r['score'],
                    'id': r['id'],
                    'title': r['title'],
                    'content': r['content']
                }
                for r in result['results']
            ]
        }
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    avg_latency = total_latency / len(queries)
    
    print(f"\n{'='*60}")
    print(f"✓ Retrieval complete!")
    print(f"  Method: {method.upper()}")
    print(f"  Total queries: {len(queries)}")
    print(f"  Avg latency: {avg_latency:.1f}ms")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")
    
    return {
        'method': method,
        'num_queries': len(queries),
        'avg_latency_ms': avg_latency,
        'output_path': output_path
    }


def main():
    """Main entry point for Phase 2 retrieval."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 2: Hybrid Retrieval with RRF-2")
    parser.add_argument("--method", type=str, default="rrf2", 
                       choices=["rrf2", "bm25", "medcpt"],
                       help="Retrieval method")
    parser.add_argument("--k", type=int, default=20, help="Number of results to retrieve")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()
    
    # Default output path
    if args.output is None:
        args.output = f"data/retrieval_results_{args.method}.json"
    
    # Initialize retriever
    retriever = HybridRetriever()
    
    # Run retrieval
    stats = run_retrieval(
        retriever=retriever,
        queries_path="data/retrieval_queries.json",
        requests_path="data/request_set.json",
        output_path=args.output,
        method=args.method,
        k=args.k
    )
    
    print(f"\n✓ Results saved to {stats['output_path']}")
    print(f"  Average latency: {stats['avg_latency_ms']:.1f}ms per query")


if __name__ == "__main__":
    main()

