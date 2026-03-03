"""Compare two RAGAS evaluation results side-by-side.

Generates:
  - Overall metrics comparison with deltas
  - Per query-type breakdown
  - Questions where pipelines disagree most
  - Failure pattern analysis
  - Console table + JSON report

Usage:
  python -m evaluation.compare_pipelines eval_basic_*.json eval_advanced_*.json
  python -m evaluation.compare_pipelines result_a.json result_b.json --output comparison.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_results(path: Path) -> dict:
    """Load evaluation results JSON."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

def compare(result_a: dict, result_b: dict) -> dict:
    """Generate comparison dict between two evaluation results.

    Returns:
        {
            "pipeline_a": "basic",
            "pipeline_b": "advanced",
            "aggregate_comparison": {
                "faithfulness": {"a": 0.85, "b": 0.91, "delta": +0.06, "winner": "b"},
                ...
            },
            "per_query_type_comparison": {
                "simple": {
                    "faithfulness": {"a": 0.90, "b": 0.95, "delta": +0.05},
                    ...
                },
                ...
            },
            "top_disagreements": [...],
            "failure_analysis": {...},
            "summary": {"a_wins": 1, "b_wins": 3, "ties": 0, "overall_winner": "b"}
        }
    """
    pipeline_a = result_a.get("pipeline", "a")
    pipeline_b = result_b.get("pipeline", "b")

    # --- Aggregate comparison ---
    ragas_a = result_a.get("ragas_scores", {})
    ragas_b = result_b.get("ragas_scores", {})

    all_metrics = sorted(set(list(ragas_a.keys()) + list(ragas_b.keys())))

    aggregate = {}
    a_wins = 0
    b_wins = 0
    ties = 0

    for metric in all_metrics:
        val_a = ragas_a.get(metric)
        val_b = ragas_b.get(metric)

        if val_a is not None and val_b is not None:
            delta = val_b - val_a
            if abs(delta) < 0.005:
                winner = "tie"
                ties += 1
            elif delta > 0:
                winner = pipeline_b
                b_wins += 1
            else:
                winner = pipeline_a
                a_wins += 1
        else:
            delta = None
            winner = None

        aggregate[metric] = {
            "a": val_a,
            "b": val_b,
            "delta": round(delta, 4) if delta is not None else None,
            "winner": winner,
        }

    # --- Per query-type comparison ---
    pqt_a = result_a.get("per_query_type", {})
    pqt_b = result_b.get("per_query_type", {})
    all_qtypes = sorted(set(list(pqt_a.keys()) + list(pqt_b.keys())))

    per_query_type_comparison = {}
    for qtype in all_qtypes:
        scores_a = pqt_a.get(qtype, {})
        scores_b = pqt_b.get(qtype, {})
        qtype_metrics = sorted(
            set(list(scores_a.keys()) + list(scores_b.keys())) - {"count"}
        )

        per_query_type_comparison[qtype] = {
            "count_a": scores_a.get("count", 0),
            "count_b": scores_b.get("count", 0),
        }
        for metric in qtype_metrics:
            va = scores_a.get(metric)
            vb = scores_b.get(metric)
            d = round(vb - va, 4) if va is not None and vb is not None else None
            per_query_type_comparison[qtype][metric] = {"a": va, "b": vb, "delta": d}

    # --- Top disagreements (questions with largest score diff) ---
    samples_a = result_a.get("per_sample", [])
    samples_b = result_b.get("per_sample", [])

    disagreements = []
    for sa, sb in zip(samples_a, samples_b):
        diffs = {}
        for metric in all_metrics:
            va = sa.get(metric)
            vb = sb.get(metric)
            if va is not None and vb is not None:
                diffs[metric] = round(vb - va, 4)

        if diffs:
            max_diff = max(abs(v) for v in diffs.values())
            disagreements.append(
                {
                    "question": sa.get("user_input", "")[:120],
                    "max_abs_diff": max_diff,
                    "diffs": diffs,
                    "synthesizer": sa.get("synthesizer_name", ""),
                }
            )

    disagreements.sort(key=lambda x: x["max_abs_diff"], reverse=True)
    top_disagreements = disagreements[:10]

    # --- Failure analysis ---
    errors_a = sum(
        1
        for s in samples_a
        if isinstance(s.get("response", ""), str) and s["response"].startswith("ERROR:")
    )
    errors_b = sum(
        1
        for s in samples_b
        if isinstance(s.get("response", ""), str) and s["response"].startswith("ERROR:")
    )

    failure_analysis = {
        f"{pipeline_a}_errors": errors_a,
        f"{pipeline_b}_errors": errors_b,
        f"{pipeline_a}_error_rate": round(errors_a / max(len(samples_a), 1), 3),
        f"{pipeline_b}_error_rate": round(errors_b / max(len(samples_b), 1), 3),
    }

    # --- Summary ---
    overall_winner = pipeline_b if b_wins > a_wins else pipeline_a if a_wins > b_wins else "tie"

    return {
        "pipeline_a": pipeline_a,
        "pipeline_b": pipeline_b,
        "aggregate_comparison": aggregate,
        "per_query_type_comparison": per_query_type_comparison,
        "top_disagreements": top_disagreements,
        "failure_analysis": failure_analysis,
        "summary": {
            "a_wins": a_wins,
            "b_wins": b_wins,
            "ties": ties,
            "overall_winner": overall_winner,
        },
    }


# ---------------------------------------------------------------------------
# Print comparison
# ---------------------------------------------------------------------------

def print_comparison(comparison: dict) -> None:
    """Print formatted comparison table to console."""
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    pa = comparison["pipeline_a"]
    pb = comparison["pipeline_b"]
    agg = comparison["aggregate_comparison"]
    summary = comparison["summary"]

    print(f"\n{'='*65}")
    print(f"  Pipeline Comparison: {pa} vs {pb}")
    print(f"{'='*65}")

    # Aggregate scores table
    if tabulate:
        headers = ["Metric", pa, pb, "Delta", "Winner"]
        rows = []
        for metric, vals in agg.items():
            a_str = f"{vals['a']:.4f}" if vals["a"] is not None else "N/A"
            b_str = f"{vals['b']:.4f}" if vals["b"] is not None else "N/A"
            d_str = f"{vals['delta']:+.4f}" if vals["delta"] is not None else "N/A"
            w_str = vals["winner"] or "N/A"
            rows.append([metric, a_str, b_str, d_str, w_str])
        print(f"\n{tabulate(rows, headers=headers, tablefmt='simple')}")
    else:
        print(f"\n  {'Metric':<28s} {pa:>8s} {pb:>8s} {'Delta':>8s} {'Winner':>10s}")
        print(f"  {'-'*62}")
        for metric, vals in agg.items():
            a_str = f"{vals['a']:.4f}" if vals["a"] is not None else "N/A"
            b_str = f"{vals['b']:.4f}" if vals["b"] is not None else "N/A"
            d_str = f"{vals['delta']:+.4f}" if vals["delta"] is not None else "N/A"
            w_str = vals["winner"] or "N/A"
            print(f"  {metric:<28s} {a_str:>8s} {b_str:>8s} {d_str:>8s} {w_str:>10s}")

    # Per query-type breakdown
    pqt = comparison.get("per_query_type_comparison", {})
    if pqt:
        print(f"\n  Per Query Type Breakdown:")
        print(f"  {'-'*50}")
        for qtype, scores in pqt.items():
            count_a = scores.pop("count_a", "?")
            count_b = scores.pop("count_b", "?")
            print(f"\n  {qtype} (n={count_a}/{count_b}):")
            for metric, vals in scores.items():
                if isinstance(vals, dict):
                    a_s = f"{vals['a']:.4f}" if vals.get("a") is not None else "N/A"
                    b_s = f"{vals['b']:.4f}" if vals.get("b") is not None else "N/A"
                    d_s = f"{vals['delta']:+.4f}" if vals.get("delta") is not None else "N/A"
                    print(f"    {metric:<26s} {a_s:>8s} {b_s:>8s} {d_s:>8s}")

    # Top disagreements
    disag = comparison.get("top_disagreements", [])
    if disag:
        print(f"\n  Top Disagreements:")
        print(f"  {'-'*50}")
        for i, d in enumerate(disag[:5], 1):
            print(f"  {i}. [{d['synthesizer'][:30]}] max_diff={d['max_abs_diff']:.4f}")
            print(f"     Q: {d['question'][:100]}")

    # Failure analysis
    fa = comparison.get("failure_analysis", {})
    if fa:
        print(f"\n  Failure Analysis:")
        for key, val in fa.items():
            print(f"    {key}: {val}")

    # Summary
    print(f"\n  {'='*40}")
    print(
        f"  Winner: {summary['overall_winner']} "
        f"({pa} wins: {summary['a_wins']}, "
        f"{pb} wins: {summary['b_wins']}, "
        f"ties: {summary['ties']})"
    )
    print(f"  {'='*40}\n")


# ---------------------------------------------------------------------------
# Save comparison
# ---------------------------------------------------------------------------

def save_comparison(comparison: dict, output_path: Path) -> None:
    """Save comparison as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    logger.info("Comparison saved to %s", output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare two RAGAS evaluation results"
    )
    parser.add_argument(
        "result_a",
        type=str,
        help="Path to first result JSON (e.g., eval_basic_*.json)",
    )
    parser.add_argument(
        "result_b",
        type=str,
        help="Path to second result JSON (e.g., eval_advanced_*.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output comparison JSON path",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    result_a = load_results(Path(args.result_a))
    result_b = load_results(Path(args.result_b))

    comparison = compare(result_a, result_b)
    print_comparison(comparison)

    if args.output:
        save_comparison(comparison, Path(args.output))
    else:
        from evaluation.config import EVAL_OUTPUT_DIR

        save_comparison(comparison, EVAL_OUTPUT_DIR / "comparison.json")


if __name__ == "__main__":
    main()
