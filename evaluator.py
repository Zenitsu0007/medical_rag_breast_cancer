"""
Medical RAG Evaluation Pipeline

Evaluates retrieval quality using:
1. Manual gold labels (when available)
2. Auto-relevance estimation (keyword-based, for fair cross-method comparison)

Metrics: Hit@k, MRR@k, Recall@k, nDCG@k (k ∈ {5, 10, 20})

Usage:
    # Evaluate single method
    python evaluator.py --results data/retrieval_results_baseline.json
    
    # Compare baseline vs RRF-2
    python evaluator.py --compare
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Any, Optional


# Question-specific keywords for auto-relevance estimation
QUESTION_KEYWORDS = {
    "q1": {
        "decisive": ["invasive ductal carcinoma", "er positive", "pr positive", "her2 negative", 
                    "lumpectomy", "sentinel lymph node", "breast conserving", "breast conservation"],
        "supportive": ["breast cancer treatment", "surgical", "mastectomy", "radiation", "ductal", 
                      "hormone receptor", "staging", "biopsy"]
    },
    "q2": {
        "decisive": ["trastuzumab", "her2", "herceptin", "her-2", "erbb2", "her2-positive", 
                    "her2 amplification", "her2 overexpression"],
        "supportive": ["targeted therapy", "breast cancer", "molecular marker", "receptor", "response"]
    },
    "q3": {
        "decisive": ["brca1", "brca2", "prophylactic mastectomy", "risk reducing", "salpingo-oophorectomy",
                    "bilateral mastectomy", "risk reduction"],
        "supportive": ["genetic mutation", "hereditary", "family history", "ovarian cancer", 
                      "cancer risk", "prevention"]
    },
    "q4": {
        "decisive": ["aromatase inhibitor", "aromatase", "androgen to estrogen", "peripheral conversion",
                    "anastrozole", "letrozole", "exemestane", "estrogen synthesis"],
        "supportive": ["postmenopausal", "hormone receptor", "endocrine therapy", "breast cancer", 
                      "estrogen", "mechanism"]
    },
    "q5": {
        "decisive": ["inflammatory breast cancer", "peau d'orange", "skin erythema", "skin edema",
                    "orange peel", "inflammatory carcinoma"],
        "supportive": ["breast cancer", "clinical features", "erythema", "edema", "skin changes", 
                      "aggressive"]
    },
    "q6": {
        "decisive": ["brca1", "triple negative", "triple-negative", "basal-like", "basal", 
                    "tnbc", "brca1 mutation"],
        "supportive": ["histological subtype", "breast cancer", "mutation", "hereditary", 
                      "receptor negative"]
    },
    "q7": {
        "decisive": ["cdk4/6 inhibitor", "cdk4", "cdk6", "palbociclib", "ribociclib", "abemaciclib",
                    "er positive metastatic", "endocrine resistance"],
        "supportive": ["metastatic breast cancer", "aromatase inhibitor", "tamoxifen", "progression",
                      "treatment strategy", "hormone therapy"]
    },
    "q8": {
        "decisive": ["sentinel lymph node", "axillary staging", "axillary lymph node dissection",
                    "slnb", "alnd", "sentinel node biopsy"],
        "supportive": ["early stage breast cancer", "lymph node", "staging", "axilla", "metastasis"]
    },
    "q9": {
        "decisive": ["paget disease", "paget's disease", "nipple", "underlying carcinoma",
                    "underlying malignancy", "dcis", "ductal carcinoma in situ"],
        "supportive": ["breast cancer", "nipple lesion", "areola", "breast malignancy"]
    },
    "q10": {
        "decisive": ["early menarche", "late menopause", "risk factor", "estrogen exposure",
                    "nulliparity", "hormone exposure", "age at menarche"],
        "supportive": ["breast cancer risk", "increased risk", "developing breast cancer",
                      "reproductive", "hormonal"]
    }
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def dcg_at_k(gains: List[int], k: int) -> float:
    dcg = 0.0
    for i, g in enumerate(gains[:k], start=1):
        if g > 0:
            dcg += g / math.log2(i + 1)
    return dcg


def idcg_at_k(gains: List[int], k: int) -> float:
    sorted_gains = sorted(gains, reverse=True)[:k]
    idcg = 0.0
    for i, g in enumerate(sorted_gains, start=1):
        if g > 0:
            idcg += g / math.log2(i + 1)
    return idcg


def estimate_relevance(qid: str, title: str, content: str) -> int:
    """Estimate relevance using question-specific keywords."""
    text = (str(title) + " " + str(content)).lower()
    keywords = QUESTION_KEYWORDS.get(qid, {"decisive": [], "supportive": []})
    
    for kw in keywords["decisive"]:
        if kw.lower() in text:
            return 2
    
    supportive_count = sum(1 for kw in keywords["supportive"] if kw.lower() in text)
    if supportive_count >= 2:
        return 1
    
    return 0


def compute_metrics(gains: List[int], k_values: List[int]) -> Dict[str, Dict[int, float]]:
    """Compute all metrics given relevance gains for ranked results."""
    num_relevant = sum(1 for g in gains if g >= 1)
    
    # Find first relevant position
    first_rel = 0
    for i, g in enumerate(gains, start=1):
        if g >= 1:
            first_rel = i
            break
    
    metrics = {"hit": {}, "mrr": {}, "recall": {}, "ndcg": {}}
    
    for k in k_values:
        # Hit@k
        metrics["hit"][k] = 1.0 if first_rel > 0 and first_rel <= k else 0.0
        
        # MRR@k
        metrics["mrr"][k] = (1.0 / first_rel) if first_rel > 0 and first_rel <= k else 0.0
        
        # Recall@k
        if num_relevant > 0:
            rel_in_topk = sum(1 for g in gains[:k] if g >= 1)
            metrics["recall"][k] = rel_in_topk / num_relevant
        else:
            metrics["recall"][k] = None
        
        # nDCG@k
        dcg = dcg_at_k(gains, k)
        idcg = idcg_at_k(gains, k)
        metrics["ndcg"][k] = dcg / idcg if idcg > 0 else None
    
    return metrics


def evaluate_results(results: Dict, gold_labels: Optional[Dict], 
                    k_values: List[int], use_auto_relevance: bool = False) -> Dict:
    """
    Evaluate retrieval results.
    
    Args:
        results: Retrieval results (per query)
        gold_labels: Manual gold labels (optional if using auto-relevance)
        k_values: List of k values for metrics
        use_auto_relevance: If True, use keyword-based relevance estimation
    """
    qids = sorted(results.keys())
    
    per_query = {}
    summary = {m: {k: [] for k in k_values} for m in ["hit", "mrr", "recall", "ndcg"]}
    latencies = []
    
    for qid in qids:
        items = results[qid].get("top_20_results", [])
        items.sort(key=lambda x: x.get("rank", 999))
        
        if use_auto_relevance:
            # Use keyword-based relevance estimation
            gains = [estimate_relevance(qid, item.get("title", ""), item.get("content", "")) 
                    for item in items]
        else:
            # Use manual gold labels
            if gold_labels is None:
                raise ValueError("gold_labels required when use_auto_relevance=False")
            
            gold_entry = gold_labels.get(qid, {})
            gold_list = gold_entry.get("manually_selected_top_5", [])
            gold_map = {x["id"]: x.get("relevance_score", 0) for x in gold_list if "id" in x}
            gains = [gold_map.get(item.get("id"), 0) for item in items]
        
        # Compute metrics
        metrics = compute_metrics(gains, k_values)
        per_query[qid] = {"metrics": metrics, "gains": gains[:20]}
        
        # Aggregate
        for m in ["hit", "mrr", "recall", "ndcg"]:
            for k in k_values:
                if metrics[m][k] is not None:
                    summary[m][k].append(metrics[m][k])
        
        # Latency
        if "latency_ms" in results[qid]:
            latencies.append(results[qid]["latency_ms"])
    
    # Compute averages
    avg_summary = {}
    for m in summary:
        avg_summary[m] = {}
        for k in k_values:
            vals = summary[m][k]
            avg_summary[m][k] = sum(vals) / len(vals) if vals else None
    
    return {
        "summary": avg_summary,
        "per_query": per_query,
        "num_queries": len(qids),
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else None
    }


def print_metrics_table(evals: Dict[str, Dict], k_values: List[int], title: str):
    """Print formatted metrics table."""
    print(f"\n{'='*70}")
    print(title)
    print('='*70)
    
    methods = list(evals.keys())
    metrics = ["hit", "mrr", "recall", "ndcg"]
    
    for metric in metrics:
        print(f"\n{metric.upper()}@k:")
        header = f"  {'Method':<25}"
        for k in k_values:
            header += f"  @{k:<6}"
        print(header)
        print("  " + "-"*55)
        
        for method in methods:
            row = f"  {method:<25}"
            for k in k_values:
                val = evals[method]["summary"][metric].get(k)
                row += f"  {val:<7.4f}" if val is not None else f"  {'N/A':<7}"
            print(row)
    
    # Latency
    print(f"\nLatency (avg per query):")
    for method in methods:
        lat = evals[method].get("avg_latency_ms")
        print(f"  {method:<25}  {lat:.1f} ms" if lat else f"  {method:<25}  N/A")


def print_improvement(base_eval: Dict, new_eval: Dict, k_values: List[int]):
    """Print improvement analysis."""
    print(f"\n{'='*70}")
    print("IMPROVEMENT ANALYSIS (Phase 2 vs Phase 1)")
    print('='*70)
    
    for metric in ["hit", "mrr", "recall", "ndcg"]:
        print(f"\n{metric.upper()}:")
        for k in k_values:
            base_val = base_eval["summary"][metric].get(k)
            new_val = new_eval["summary"][metric].get(k)
            
            if base_val is not None and new_val is not None:
                delta = new_val - base_val
                pct = (delta / base_val * 100) if base_val > 0 else 0
                sign = "+" if delta >= 0 else ""
                print(f"  @{k}: {base_val:.4f} → {new_val:.4f} ({sign}{delta:.4f}, {sign}{pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Medical RAG Evaluation")
    parser.add_argument("--results", type=Path, help="Single result file to evaluate")
    parser.add_argument("--gold", type=Path, default=Path("data/manual_labels.json"))
    parser.add_argument("--compare", action="store_true", help="Compare baseline vs RRF-2")
    parser.add_argument("--auto-relevance", action="store_true", 
                       help="Use keyword-based auto-relevance (for fair cross-method comparison)")
    parser.add_argument("--out", type=Path, default=Path("data/evaluation_report.json"))
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    args = parser.parse_args()
    
    k_values = args.k
    
    if args.compare:
        # Compare baseline vs RRF-2
        print("\n" + "="*70)
        print("MEDICAL RAG EVALUATION - Phase 1 vs Phase 2 Comparison")
        print("="*70)
        
        baseline_path = Path("data/retrieval_results_baseline.json")
        rrf2_path = Path("data/retrieval_results_rrf2.json")
        
        if not baseline_path.exists() or not rrf2_path.exists():
            print("Error: Missing result files. Run retriever.py first.")
            return
        
        baseline_results = load_json(baseline_path)
        rrf2_results = load_json(rrf2_path)
        gold_labels = load_json(args.gold) if args.gold.exists() else None
        
        # Evaluate with auto-relevance (fair comparison)
        print("\n[Using Auto-Relevance Estimation for Fair Comparison]")
        
        baseline_eval = evaluate_results(baseline_results, gold_labels, k_values, use_auto_relevance=True)
        rrf2_eval = evaluate_results(rrf2_results, gold_labels, k_values, use_auto_relevance=True)
        
        evals = {
            "Phase1-MedCPT (Baseline)": baseline_eval,
            "Phase2-RRF2 (BM25+MedCPT)": rrf2_eval
        }
        
        print_metrics_table(evals, k_values, "RETRIEVAL QUALITY COMPARISON")
        print_improvement(baseline_eval, rrf2_eval, k_values)
        
        # Save report
        report = {
            "evaluation_method": "auto_relevance",
            "k_values": k_values,
            "baseline": {
                "name": "Phase1-MedCPT",
                "summary": baseline_eval["summary"],
                "avg_latency_ms": baseline_eval.get("avg_latency_ms")
            },
            "rrf2": {
                "name": "Phase2-RRF2",
                "summary": rrf2_eval["summary"],
                "avg_latency_ms": rrf2_eval.get("avg_latency_ms")
            }
        }
        save_json(args.out, report)
        print(f"\n✓ Report saved to {args.out}")
        
    elif args.results:
        # Evaluate single result file
        results = load_json(args.results)
        gold_labels = load_json(args.gold) if args.gold.exists() else None
        
        eval_result = evaluate_results(
            results, gold_labels, k_values, 
            use_auto_relevance=args.auto_relevance
        )
        
        method = "Auto-Relevance" if args.auto_relevance else "Gold Labels"
        print(f"\n=== Evaluation ({method}) ===")
        print(f"Results file: {args.results}")
        print(f"Queries: {eval_result['num_queries']}")
        
        for m in ["hit", "mrr", "recall", "ndcg"]:
            vals = [f"{m}@{k}={eval_result['summary'][m][k]:.4f}" 
                   for k in k_values if eval_result['summary'][m][k] is not None]
            print("  " + "  ".join(vals))
        
        if eval_result.get("avg_latency_ms"):
            print(f"  Avg latency: {eval_result['avg_latency_ms']:.1f}ms")
        
        save_json(args.out, eval_result)
        print(f"\n✓ Report saved to {args.out}")
    
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python evaluator.py --compare                    # Compare baseline vs RRF-2")
        print("  python evaluator.py --results data/retrieval_results_baseline.json")
        print("  python evaluator.py --results data/retrieval_results_rrf2.json --auto-relevance")


if __name__ == "__main__":
    main()
