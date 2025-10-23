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
⏳ Evaluator: To be implemented

---

## TODO

### 评估器实现 (Evaluator Implementation)

需要实现 `evaluator.py` 来量化检索系统的质量。

#### 数据说明

**1. Raw Retrieval Results（原始检索结果）**
- 文件：`data/raw_retrieval_results.json`
- 内容：每个问题的 top-20 检索结果
- 结构：每个文档有 `rank`（1-20的检索排名）、`score`（相似度分数）、`id`、`title`、`content`

**2. Manual Labels（人工标注 - 黄金标准）**
- 文件：`data/manual_labels.json`
- 内容：每个问题人工标注的 top-5 相关文档
- **重要**：每个文档有 `relevance_score` 相关度分数：
  - **2 = Decisive（决定性）** - 直接回答问题或强力支持答案
  - **1 = Helpful（有帮助）** - 提供相关背景知识但不直接回答
  - **0 = Not relevant（不相关）** - 虽然被检索到但对该问题无用
- **注意**：`rank` 字段是该文档在**原始检索结果中的位置**（1-20），不是人工选择的顺序

#### 如何使用 Manual Labels 作为黄金标准

**判断文档是否相关：**
```python
# 方法1：二元相关性（binary relevance）
# 相关文档 = relevance_score >= 1 的文档
relevant_ids = [doc['id'] for doc in manual_labels[qid]['manually_selected_top_5'] 
                if doc['relevance_score'] >= 1]

# 方法2：使用相关度等级（graded relevance，用于nDCG）
# 保留每个文档的相关度分数
relevant_docs = {doc['id']: doc['relevance_score'] 
                 for doc in manual_labels[qid]['manually_selected_top_5']}
```

**为什么保留原始 rank？**
- Rank 表示文档在检索系统中的位置（1-20）
- 用于计算指标时，需要知道相关文档出现在检索结果的哪个位置
- 例如：一个 `relevance_score=2` 的文档在 `rank=15`，说明系统把高度相关的文档排在了第15位

#### 需要实现的指标（k = 5, 10, 20）

**1. Recall@k（召回率）**

在前 k 个检索结果中找到了多少比例的相关文档

```python
# 相关文档定义：relevance_score >= 1
Recall@k = (前k个结果中相关文档数量) / (总相关文档数量)
```

**2. nDCG@k（归一化折损累积增益）**

考虑排序位置和相关度等级的检索质量指标

```python
# 使用 relevance_score (0/1/2) 作为相关度等级
DCG@k = Σ (relevance_score / log2(position+1)) for position=1 to k
IDCG@k = 理想排序的DCG（相关文档按relevance_score降序排列）
nDCG@k = DCG@k / IDCG@k
```

**3. MRR@k（平均倒数排名）**

第一个相关文档（`relevance_score >= 1`）出现位置的倒数

```python
MRR@k = Average(1 / rank_first_relevant) 仅考虑前k个结果
# 如果前k个结果中没有相关文档，则该查询的MRR = 0
```

**4. Hit@k（命中率）**

前 k 个结果中至少包含一个相关文档的查询比例

```python
Hit@k = (至少有1个相关文档在前k的查询数) / (总查询数)
```

#### 输出要求

- 创建 `evaluator.py` 文件
- 生成 `data/evaluation_report.json` 包含所有指标
- 打印评估报告（每个指标在 k=5, 10, 20 的平均值）

#### 实现示例

```python
import json

# 加载数据
with open('data/raw_retrieval_results.json') as f:
    raw_results = json.load(f)
with open('data/manual_labels.json') as f:
    manual_labels = json.load(f)

# 对每个问题计算指标
for qid in raw_results.keys():
    # 获取检索结果ID（按rank 1-20排序）
    retrieved_ids = [r['id'] for r in raw_results[qid]['top_20_results']]
    
    # 获取相关文档ID（relevance_score >= 1）
    relevant_ids = [doc['id'] for doc in manual_labels[qid]['manually_selected_top_5']
                    if doc['relevance_score'] >= 1]
    
    # 获取相关文档的relevance_score（用于nDCG）
    relevance_dict = {doc['id']: doc['relevance_score'] 
                      for doc in manual_labels[qid]['manually_selected_top_5']}
    
    # 计算 Recall@5
    retrieved_top5 = retrieved_ids[:5]
    relevant_in_top5 = len(set(retrieved_top5) & set(relevant_ids))
    recall_at_5 = relevant_in_top5 / len(relevant_ids) if relevant_ids else 0
    
    # 计算 nDCG@5 (需要考虑relevance_score)
    # ...你的实现...
```

