from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CASES = (
    PROJECT_DIR / "eval_results" / "test_cases_20260405_225548.json"
)
LABELS = {"FULLY_SUPPORTED", "PARTIALLY_SUPPORTED", "NO_INFORMATION"}
EVIDENCE_STATUSES = {"NEEDS_REVIEW", "ORGANIZATION_EVIDENCE_REQUIRED", "INSUFFICIENT_CONTEXT",
                     "CLARIFICATION_REQUIRED", "INVALID_CITATIONS"}


def load_test_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("cases")
    if not isinstance(data, list) or not data:
        raise ValueError("Test cases must be a non-empty JSON array.")

    seen = set()
    for number, case in enumerate(data, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Case {number}: expected a JSON object.")
        query = case.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Case {number}: query must be a non-empty string.")
        key = " ".join(query.split()).casefold()
        if key in seen:
            raise ValueError(f"Case {number}: duplicate query.")
        seen.add(key)

        expected = case.get("expected")
        if expected is not None and (
            not isinstance(expected, str) or expected not in LABELS
        ):
            raise ValueError(f"Case {number}: invalid expected label.")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a fixed question set against the current RAG pipeline."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--index-dir", type=Path, default=PROJECT_DIR / "faiss_index_full_baseline")
    parser.add_argument("--model", default="llama3")
    parser.add_argument(
        "--score-reviewed",
        action="store_true",
        help="Compare manually reviewed expected_evidence_status values (not answer accuracy).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_DIR / "eval_results"
    )
    args = parser.parse_args()

    cases = load_test_cases(args.cases)
    if args.score_reviewed and any(
        case.get("expected_evidence_status") not in EVIDENCE_STATUSES
        or case.get("review_status") != "reviewed"
        for case in cases
    ):
        parser.error("Scoring requires expected_evidence_status and review_status='reviewed' for every case. Old similarity-based labels are not evidence labels.")

    from generator import RAGPipeline

    pipeline = RAGPipeline(model=args.model, index_dir=args.index_dir)
    results = []

    for number, case in enumerate(cases, start=1):
        print(f"[{number:02d}/{len(cases)}] Processing: {case['query']}", flush=True)
        started = time.perf_counter()
        response = pipeline.query(case["query"])
        elapsed = time.perf_counter() - started
        actual = response.evidence_status.value
        match = (
            actual == case["expected_evidence_status"] if args.score_reviewed else None
        )

        results.append({
            "query": case["query"],
            "reference_evidence_status": case.get("expected_evidence_status"),
            "evidence_status": actual,
            "retrieval_signal": response.retrieval_signal.value,
            "question_intent": response.question_intent,
            "cited_control_ids": response.cited_control_ids,
            "invalid_citations": response.invalid_citations,
            "match": match,
            "top_score": response.top_score,
            "query_seconds": round(elapsed, 3),
            "answer": response.answer,
            "retrieved": [
                {"chunk_id": doc.chunk_id, "control_id": doc.control_id,
                 "score": doc.score, "text": doc.text,
                 "is_withdrawn": doc.is_withdrawn, "withdrawal_notice": doc.withdrawal_notice}
                for doc in response.retrieved
            ],
        })
        print(
            f"[{number:02d}/{len(cases)}] evidence={actual} retrieval={response.retrieval_signal.value} "
            f"score={response.top_score:.4f} time={elapsed:.2f}s"
        )

    correct = (
        sum(row["match"] for row in results)
        if args.score_reviewed else None
    )
    report = {
        "report_schema_version": 2,
        "code_sha256": {name: hashlib.sha256((PROJECT_DIR / name).read_bytes()).hexdigest()
                        for name in ("generator.py", "retriever.py", "ingestion.py", "evaluate.py")},
        "index_dir": str(args.index_dir.resolve()),
        "index_sha256": hashlib.sha256((args.index_dir / "index.bin").read_bytes()).hexdigest(),
        "index_manifest": pipeline._retriever.manifest,
        "generation_model": args.model,
        "test_cases_path": str(args.cases.resolve()),
        "test_cases_sha256": hashlib.sha256(
            args.cases.read_bytes()
        ).hexdigest(),
        "labels_reviewed_for_current_corpus": args.score_reviewed,
        "test_cases": cases,
        "total": len(results),
        "correct": correct,
        "evidence_status_agreement": (
            correct / len(results) if correct is not None else None
        ),
        "results": results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"fixed_eval_{time.time_ns()}.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if correct is None:
        print("[INFO] Predictions saved; unreviewed labels were not scored.")
    else:
        print(
            f"Evidence-status agreement (not answer accuracy): {correct}/{len(results)} "
            f"({correct / len(results):.1%})"
        )
    print(f"[INFO] Saved: {output_path}")


if __name__ == "__main__":
    main()
