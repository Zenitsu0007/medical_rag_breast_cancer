# Medical RAG System for Breast Cancer Domain

**Team 13: Zejia Shen, Zilu Li, Nerako Li**

**Domain**: Healthcare (Breast Cancer)

## Overview

A domain-specific Retrieval-Augmented Generation (RAG) system for breast cancer medical question-answering. The system implements and compares two retrieval approaches:

- **Phase 1 (Baseline)**: MedCPT semantic retrieval with FAISS
- **Phase 2 (Improved)**: Hybrid BM25 + MedCPT with Reciprocal Rank Fusion (RRF-2)

Based on the [MedRAG paper](https://aclanthology.org/2024.findings-acl.372/) (Xiong et al., ACL 2024).

## Results Summary

| Metric | Phase 1 (MedCPT) | Phase 2 (RRF-2) | Improvement |
|--------|-----------------|-----------------|-------------|
| Recall@5 | 0.2983 | 0.3397 | **+13.9%** |
| Recall@10 | 0.5081 | 0.5955 | **+17.2%** |
| nDCG@5 | 0.6118 | 0.6932 | **+13.3%** |
| nDCG@10 | 0.6089 | 0.7207 | **+18.4%** |
| Latency | N/A | 1584ms | - |

## Repository Structure

```
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup_datasets.py         # Step 1: Download and prepare datasets
├── build_index.py            # Step 2: Build MedCPT embeddings + FAISS index
├── retriever.py              # Step 3: Run retrieval (supports baseline & RRF-2)
├── evaluator.py              # Step 4: Evaluate retrieval quality
└── data/
    ├── combined_corpus.jsonl           # Combined corpus (135,360 docs)
    ├── pubmed_breast_cancer_corpus.jsonl  # PubMed abstracts (9,513)
    ├── textbooks_corpus.jsonl          # Medical textbooks (125,847)
    ├── request_set.json                # 10 test questions with answers
    ├── retrieval_queries.json          # Questions for retrieval
    ├── manual_labels.json              # Gold labels with relevance scores
    ├── faiss_index.index               # FAISS vector index
    ├── faiss_index_embeddings.npy      # Document embeddings
    ├── retrieval_results_baseline.json # Phase 1 results
    └── retrieval_results_rrf2.json     # Phase 2 results
```

## Quick Start

### Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Evaluation (Pre-built Index)

```bash
# Compare Phase 1 vs Phase 2
python evaluator.py --compare
```

### Full Pipeline (From Scratch)

```bash
# Step 1: Download datasets (~5 min)
python setup_datasets.py

# Step 2: Build FAISS index (~1.5 hours on M1 Mac)
python build_index.py

# Step 3: Run retrieval
python retriever.py --method medcpt --output data/retrieval_results_baseline.json
python retriever.py --method rrf2 --output data/retrieval_results_rrf2.json

# Step 4: Evaluate
python evaluator.py --compare
```

## Dataset

**Combined Corpus**: 135,360 documents from two sources

| Source | Documents | Description |
|--------|-----------|-------------|
| Medical Textbooks | 125,847 | 18 USMLE reference textbooks (from MedRAG) |
| PubMed Abstracts | 9,513 | Breast cancer research abstracts |

**Request Set**: 10 clinically-grounded multiple-choice questions covering:
- Diagnosis (invasive ductal carcinoma, inflammatory breast cancer, Paget disease)
- Molecular markers (HER2, BRCA1)
- Treatment (aromatase inhibitors, CDK4/6 inhibitors, trastuzumab)
- Staging (sentinel lymph node biopsy)
- Risk factors (hormonal exposure)

## System Components

### 1. Chunking Strategy

Our chunking strategy follows the MedRAG approach, with source-specific handling:

**Medical Textbooks**: The textbooks corpus (from MedRAG) contains 18 widely-used USMLE reference textbooks. Documents are pre-chunked to a maximum of 1000 characters using LangChain's `RecursiveCharacterTextSplitter`. This chunking preserves semantic coherence while ensuring each snippet is small enough for effective embedding and retrieval. The chunking was performed by the MedRAG team and we use their pre-processed corpus directly.

**PubMed Abstracts**: PubMed abstracts are used as-is without additional chunking. Each abstract (title + abstract text) forms a single retrieval unit. This is appropriate because:
- Abstracts are already concise summaries (typically 200-300 words)
- They are self-contained units of information
- Splitting abstracts would break semantic coherence

| Source | Chunking | Avg. Length | Rationale |
|--------|----------|-------------|-----------|
| Textbooks | RecursiveCharacterTextSplitter (≤1000 chars) | 182 tokens | Long chapters need splitting |
| PubMed | None (full abstracts) | 296 tokens | Already concise and coherent |

### 2. Embedding Generation and Storage

**Embedding Model**: We use [MedCPT](https://github.com/ncbi/MedCPT) (Medical Contrastive Pre-trained Transformers), a biomedical embedding model trained on 255 million PubMed search logs using contrastive learning.

- **Document Encoder**: `ncbi/MedCPT-Article-Encoder` - optimized for encoding biomedical article content
- **Query Encoder**: `ncbi/MedCPT-Query-Encoder` - optimized for encoding information-seeking queries
- **Embedding Dimension**: 768

**Generation Process** (`build_index.py`):
1. Load all 135,360 documents from combined corpus
2. Encode each document using MedCPT Article Encoder (batch size=16)
3. Process with checkpointing every 50 batches for fault tolerance
4. L2-normalize embeddings for cosine similarity search

**Storage**:
- `faiss_index_embeddings.npy`: Raw embeddings (135,360 × 768 float32 = ~397MB)
- `faiss_index.index`: FAISS IndexFlatIP index (~397MB)

### 3. Retrieval Mechanism

We implement two retrieval mechanisms:

**Phase 1 - Dense Vector Search (MedCPT + FAISS)**:
- Query is encoded using MedCPT Query Encoder
- FAISS IndexFlatIP performs exact inner product search (cosine similarity after L2 normalization)
- Returns top-k documents ranked by semantic similarity

**Phase 2 - Hybrid Search (BM25 + MedCPT + RRF)**:
- **BM25**: Lexical retrieval using `rank_bm25` library (BM25Okapi variant)
  - Tokenization: lowercase + alphanumeric word extraction
  - Parameters: k1=1.5, b=0.75 (defaults)
- **MedCPT**: Dense retrieval as in Phase 1
- **Fusion**: Reciprocal Rank Fusion combines both rankings
  - RRF score = Σ 1/(k + rank) where k=60
  - Documents ranked highly by both methods get boosted scores

```
┌─────────────┐     ┌─────────────┐
│   Query     │     │   Query     │
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
┌─────────────┐     ┌─────────────┐
│    BM25     │     │   MedCPT    │
│  (Lexical)  │     │ (Semantic)  │
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
┌─────────────┐     ┌─────────────┐
│  Top-100    │     │  Top-100    │
│  Rankings   │     │  Rankings   │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └───────┬───────────┘
               ▼
       ┌─────────────┐
       │  RRF Fusion │
       │   (k=60)    │
       └──────┬──────┘
               │
               ▼
       ┌─────────────┐
       │   Top-20    │
       │   Results   │
       └─────────────┘
```

## Evaluation

### Metrics

- **Hit@k**: Whether relevant document appears in top-k
- **MRR@k**: Reciprocal rank of first relevant document
- **Recall@k**: Fraction of relevant documents in top-k
- **nDCG@k**: Normalized Discounted Cumulative Gain (graded relevance: 2=decisive, 1=supportive, 0=irrelevant)

### Manual Labeling Protocol

For each question:
1. Retrieved top-20 documents using MedCPT
2. Manually reviewed all 20 documents
3. Selected top-5 most relevant with graded relevance scores:
   - **2 = Decisive**: Directly answers the question
   - **1 = Supportive**: Provides relevant context
   - **0 = Not relevant**: Not useful for the question

### Auto-Relevance Estimation

For fair cross-method comparison, we use keyword-based auto-relevance estimation:
- Question-specific decisive keywords (score=2 if matched)
- Supportive keywords (score=1 if ≥2 matched)

This avoids bias from gold labels created only from Phase 1 results.

## Key Findings

1. **RRF-2 improves ranking quality**: +13-18% improvement in Recall and nDCG by combining lexical and semantic retrieval.

2. **Textbook content is highly valuable**: RRF-2 retrieves authoritative textbook passages (Surgery_Schwartz, Pathology_Robbins, etc.) that directly answer clinical questions.

3. **Trade-off**: RRF-2 has higher latency (~1.6s) due to running two retrievers, but the quality improvement justifies this for medical applications where accuracy is critical.

## References

- Xiong et al., "[Benchmarking Retrieval-Augmented Generation for Medicine](https://aclanthology.org/2024.findings-acl.372/)", ACL Findings 2024
- [MedRAG Toolkit](https://github.com/Teddy-XiongGZ/MedRAG)
- [MedCPT](https://github.com/ncbi/MedCPT) - Biomedical embeddings
