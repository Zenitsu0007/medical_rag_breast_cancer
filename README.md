# Medical RAG System for Breast Cancer Domain (Phase 1)

**Team 13: Zejia Shen, Zilu Li, Nerako Li**

**Domain**: Healthcare (Breast Cancer)

## Overview

Medical RAG system for breast cancer question-answering using biomedical textbooks and PubMed abstracts as corpus for retrieval.

## Files

### Python Scripts

- **`setup_datasets.py`** - Downloads and formats the two corpora (textbooks and PubMed abstracts)
- **`build_retriever_safe.py`** - Encodes all 135K documents using MedCPT and builds FAISS index (~1.5 hours)
- **`finalize_index.py`** - Finalizes FAISS index from checkpoint embeddings
- **`perform_retrieval.py`** - Retrieves top-20 results per question and manually selects top-5 most relevant

### Data Files

**Datasets:**
- **`data/textbooks_corpus.jsonl`** - 125,847 snippets from 18 medical textbooks
- **`data/pubmed_breast_cancer_corpus.jsonl`** - 9,513 breast cancer research abstracts
- **`data/combined_corpus.jsonl`** - Combined corpus of 135,360 documents

**Request Set:**
- **`data/request_set.json`** - 10 multiple-choice questions with options and answers
- **`data/retrieval_queries.json`** - Questions only (for retrieval)

**Retrieval Results:**
- **`data/raw_retrieval_results.json`** - Top-20 retrieved documents per question
- **`data/manual_labels.json`** - Manually selected top-5 relevant documents per question
- **`data/raw_top5_results.json`** - System's top-5 results (for comparison)

**Index:**
- **`data/faiss_index.index`** - FAISS vector index (397MB)
- **`data/faiss_index_embeddings.npy`** - Document embeddings (397MB)

### Configuration

- **`requirements.txt`** - Python dependencies
- **`.gitignore`** - Git ignore rules (excludes large data files)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Manual Labeling Methodology

For each of the 10 questions, we:

1. **Retrieved top-20 documents** using MedCPT + FAISS
2. **Manually reviewed** all 20 documents by analyzing:
   - How well the content addresses the specific question
   - Presence of key medical terms and concepts
   - Quality and specificity of information
3. **Selected top-5 most relevant** documents from the 20 results
4. **Assigned relevance scores** to each selected document:
   - **2 = Decisive** - Directly answers or strongly supports the answer (e.g., explains mechanism, describes exact condition)
   - **1 = Helpful** - Provides relevant context but doesn't directly answer (e.g., discusses related concepts, mentions in passing)
   - **0 = Not relevant** - Retrieved but not actually useful for this question

**Note**: The `rank` field indicates where the document appeared in the **original retrieval results** (1-20), NOT the manual selection order. This is crucial for calculating retrieval metrics.

## Completed

✅ Dataset: 135,360 medical documents (2 types)  
✅ Request Set: 10 breast cancer questions  
✅ Skeleton Retrieval System: MedCPT + FAISS retriever  
✅ Manual Labels: Top-5 docs/question with relevance scores (0/1/2)  
✅ Raw Results: Top-20 retrieval results per question  
✅ Evaluator: Evaluate on benchmark（Hit/MRR/Recall/nDCG@5/10/20）

---
