import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def log2(x: float) -> float:
    return math.log2(x)

def dcg_at_k(retrieved_ids: List[str], gains: Dict[str, int], k: int) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        g = gains.get(doc_id, 0)
        if g > 0:
            dcg += g / log2(i + 1)
    return dcg

def idcg_at_k(gold_gains: List[int], k: int) -> float:
    if not gold_gains:
        return 0.0
    sorted_g = sorted(gold_gains, reverse=True)[:k]
    idcg = 0.0
    for i, g in enumerate(sorted_g, start=1):
        if g > 0:
            idcg += g / log2(i + 1)
    return idcg

def first_relevant_rank(retrieved_ids: List[str], gains: Dict[str, int], k: int) -> int:
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if gains.get(doc_id, 0) >= 1:
            return i
    return 0

def metrics_for_query(
    retrieved_ids: List[str],
    gold_gains_map: Dict[str, int],
    k_values: List[int]
) -> Dict[str, Dict[int, float]]:
    gold_gains = list(gold_gains_map.values())
    gold_relevant_count = sum(1 for g in gold_gains if g >= 1)
    has_relevant_gold = gold_relevant_count > 0

    out = {
        "hit": {},
        "mrr": {},
        "recall": {},
        "ndcg": {},
        "flags": {"has_relevant_gold": has_relevant_gold, "gold_rels": gold_relevant_count}
    }

    for k in k_values:
        fr = first_relevant_rank(retrieved_ids, gold_gains_map, k)
        out["hit"][k] = 1.0 if fr > 0 else 0.0
        out["mrr"][k] = (1.0 / fr) if fr > 0 else 0.0

        if has_relevant_gold:
            retrieved_topk = retrieved_ids[:k]
            bin_relevant_in_topk = sum(1 for d in retrieved_topk if gold_gains_map.get(d, 0) >= 1)
            out["recall"][k] = bin_relevant_in_topk / gold_relevant_count
        else:
            out["recall"][k] = None

        dcg = dcg_at_k(retrieved_ids, gold_gains_map, k)
        idcg = idcg_at_k(gold_gains, k)
        if idcg > 0:
            out["ndcg"][k] = dcg / idcg
        else:
            out["ndcg"][k] = None

    return out

def aggregate_all(per_query: Dict[str, Dict[str, Dict[int, float]]], k_values: List[int]) -> Dict[str, Dict[int, float]]:
    summary: Dict[str, Dict[int, float]] = {m: {k: 0.0 for k in k_values} for m in ["hit", "mrr", "recall", "ndcg"]}
    counts: Dict[str, Dict[int, int]] = {m: {k: 0 for k in k_values} for m in ["hit", "mrr", "recall", "ndcg"]}

    for qid, met in per_query.items():
        for k in k_values:
            summary["hit"][k] += met["hit"][k]
            counts["hit"][k] += 1

            summary["mrr"][k] += met["mrr"][k]
            counts["mrr"][k] += 1

            if met["recall"][k] is not None:
                summary["recall"][k] += met["recall"][k]
                counts["recall"][k] += 1

            if met["ndcg"][k] is not None:
                summary["ndcg"][k] += met["ndcg"][k]
                counts["ndcg"][k] += 1

    for m in summary:
        for k in summary[m]:
            c = counts[m][k]
            summary[m][k] = (summary[m][k] / c) if c > 0 else None

    summary["_counts"] = counts
    return summary

def read_retrieved_ids(raw_entry: Dict[str, Any]) -> List[str]:
    items = raw_entry.get("top_20_results", [])
    items = [x for x in items if "id" in x and "rank" in x]
    items.sort(key=lambda x: x["rank"])
    return [x["id"] for x in items]

def read_gold_gains(gold_entry: Dict[str, Any]) -> Dict[str, int]:
    lst = gold_entry.get("manually_selected_top_5", [])
    out = {}
    for x in lst:
        if "id" in x and "relevance_score" in x:
            try:
                out[x["id"]] = int(x["relevance_score"])
            except Exception:
                try:
                    out[x["id"]] = int(str(x["relevance_score"]).strip())
                except Exception:
                    out[x["id"]] = 0
    return out

def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality with Hit/MRR/Recall/nDCG at k.")
    parser.add_argument("--raw", type=Path, default=Path("data/raw_retrieval_results.json"), help="Path to raw retrieval JSON.")
    parser.add_argument("--gold", type=Path, default=Path("data/manual_labels.json"), help="Path to manual labels JSON.")
    parser.add_argument("--out", type=Path, default=Path("data/evaluation_report.json"), help="Path to write summary JSON.")
    parser.add_argument("--per-query", type=Path, default=None, help="Optional path to write per-query metrics JSONL.")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20], help="List of cutoff k values.")
    args = parser.parse_args()

    raw_results = load_json(args.raw)
    manual_labels = load_json(args.gold)

    qids = sorted(set(raw_results.keys()) & set(manual_labels.keys()))

    per_query_metrics: Dict[str, Dict[str, Dict[int, float]]] = {}
    skipped_no_gold = 0
    sanity_logs: List[str] = []

    for qid in qids:
        retrieved_ids = read_retrieved_ids(raw_results[qid])
        gold_gains_map = read_gold_gains(manual_labels[qid])

        if not retrieved_ids:
            sanity_logs.append(f"[WARN] qid={qid} has empty retrieved list.")
        if not gold_gains_map:
            skipped_no_gold += 1

        per_query_metrics[qid] = metrics_for_query(
            retrieved_ids=retrieved_ids,
            gold_gains_map=gold_gains_map,
            k_values=args.k
        )

    summary = aggregate_all(per_query_metrics, args.k)

    print("=== Retrieval Evaluation Summary ===")
    print(f"#queries (intersection) = {len(qids)}")
    if skipped_no_gold > 0:
        print(f"Note: {skipped_no_gold} queries have no relevant gold; recall/ndcg skip those queries.")
    for m in ["hit", "mrr", "recall", "ndcg"]:
        row = []
        for k in args.k:
            v = summary[m][k]
            row.append(f"{m}@{k}={v:.4f}" if v is not None else f"{m}@{k}=NA")
        print("  " + "  ".join(row))

    report = {
        "meta": {
            "num_queries_intersection": len(qids),
            "skipped_no_gold": skipped_no_gold,
            "k_values": args.k
        },
        "summary": summary
    }
    save_json(args.out, report)

    if args.per_query:
        args.per_query.parent.mkdir(parents=True, exist_ok=True)
        with args.per_query.open("w", encoding="utf-8") as f:
            for qid in qids:
                row = {"qid": qid, **per_query_metrics[qid]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    sanity_path = Path("data/eval_sanity_log.txt")
    if sanity_logs:
        sanity_path.parent.mkdir(parents=True, exist_ok=True)
        sanity_path.write_text("\n".join(sanity_logs), encoding="utf-8")

if __name__ == "__main__":
    main()
