"""Writes the evaluation report (JSON + Markdown) and prints a short summary."""
from __future__ import annotations

import json
from pathlib import Path

from .runner import LOWER_IS_BETTER, THRESHOLDS

LABELS = {
    "retrieval.hit_rate": "Hit rate@k (correct source in top k)",
    "retrieval.mrr": "MRR (how high the correct source ranks)",
    "retrieval.precision": "Precision@k (share of sources that are correct)",
    "retrieval.context_recall": "Context recall (key facts in the sources)",
    "retrieval.unanswerable_blocked": "Unanswerable: nothing passed the relevance check",
    "generation.answer_rate": "Answer rate (answerable questions answered)",
    "generation.false_refusal_rate": "False refusal rate (said 'no information' wrongly)",
    "generation.fact_recall": "Fact recall (key facts in the answer)",
    "generation.citation_valid_rate": "Citation valid rate",
    "generation.citation_precision": "Citation precision (cited sources are correct)",
    "generation.refusal_accuracy": "Refusal accuracy (unanswerable -> 'no information')",
}


def _delta(name: str, value, previous: dict | None) -> str:
    if previous is None or value is None:
        return ""
    old = previous.get("metrics", {}).get(name)
    if old is None:
        return ""
    change = value - old
    if abs(change) < 0.0005:
        return "="
    better = change < 0 if name in LOWER_IS_BETTER else change > 0
    return f"{'▲' if change > 0 else '▼'} {change:+.3f} {'better' if better else 'worse'}"


def write_reports(summary: dict, failed: list[str], previous: dict | None, folder: Path) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    summary = {**summary, "thresholds_failed": failed, "passed": not failed}
    json_file = folder / f"eval-{summary['run_id']}.json"
    json_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    lines = [f"# RAG evaluation · {summary['run_id']}", "",
             f"**Result: {'PASS' if not failed else 'FAIL'}** · {summary['questions']} questions · "
             f"errors: {summary['errors']} · mean {summary['latency']['mean_s']}s per question", "",
             "| Setting | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary["settings"].items()]
    lines += ["", "## Metrics", "", "| Metric | Score | Threshold | vs previous |", "|---|---|---|---|"]
    for name, value in summary["metrics"].items():
        limit = THRESHOLDS.get(name)
        limit_text = "" if limit is None else (f"≤ {limit}" if name in LOWER_IS_BETTER else f"≥ {limit}")
        shown = "n/a" if value is None else f"{value:.3f}"
        lines.append(f"| {LABELS.get(name, name)} | {shown} | {limit_text} | {_delta(name, value, previous)} |")

    lines += ["", "## Retrieval misses by cause", ""]
    lines += [f"- {cause}: {n}" for cause, n in summary["diagnosis"].items()] or ["- none"]

    lines += ["", "## By question type", "", "| Tag | Questions | Hit rate | Fact recall | Refusal accuracy |",
              "|---|---|---|---|---|"]
    for tag, row in summary["by_tag"].items():
        cells = ["-" if row[k] is None else f"{row[k]:.2f}" for k in ("hit_rate", "fact_recall", "refusal_accuracy")]
        lines.append(f"| {tag} | {row['questions']} | " + " | ".join(cells) + " |")

    lines += ["", "## Questions to look at", ""]
    if not summary["failures"]:
        lines.append("None - every question passed.")
    for f in summary["failures"]:
        lines.append(f"### {f['id']}\n\n**Q:** {f['question']}\n")
        lines += [f"- {reason}" for reason in f["reasons"]]
        if f.get("answer"):
            lines.append(f"\n**Answer:** {f['answer']}")
        if f.get("sources"):
            lines.append("\n**Sources sent:** " + "; ".join(f["sources"]))
        lines.append("")
    if failed:
        lines += ["## Thresholds not met", ""] + [f"- {x}" for x in failed]

    md_file = folder / f"eval-{summary['run_id']}.md"
    md_file.write_text("\n".join(lines) + "\n")
    return json_file, md_file


def print_summary(summary: dict, failed: list[str], previous: dict | None) -> None:
    print("\n" + "=" * 78)
    print(f"{'Metric':54s} {'Score':>7s}  vs previous")
    print("-" * 78)
    for name, value in summary["metrics"].items():
        shown = "  n/a" if value is None else f"{value:7.3f}"
        print(f"{LABELS.get(name, name)[:54]:54s} {shown}  {_delta(name, value, previous)}")
    print("-" * 78)
    if summary["diagnosis"]:
        print("Retrieval misses: " + ", ".join(f"{k} {v}" for k, v in summary["diagnosis"].items()))
    print(f"Questions to look at: {len(summary['failures'])}   errors: {summary['errors']}")
    print(f"\nResult: {'PASS' if not failed else 'FAIL'}")
    for item in failed:
        print(f"  ✗ {item}")
