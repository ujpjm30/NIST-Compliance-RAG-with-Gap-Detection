from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict

import ollama

from generator import RAGPipeline, SupportLevel

OLLAMA_MODEL = "llama3"
RESULTS_DIR  = "eval_results"

@dataclass
class TestCase:
    query:     str
    expected:  str   # "FULLY_SUPPORTED" | "PARTIALLY_SUPPORTED" | "NO_INFORMATION"
    rationale: str

@dataclass
class EvalResult:
    query:          str
    expected:       str
    actual:         str
    top_score:      float
    match:          bool
    answer_preview: str   # first 150 chars


TESTGEN_SYSTEM = """\
You are designing evaluation test cases for a NIST SP 800-53 RAG system.

The knowledge base contains controls from THESE families ONLY:
  AC (Access Control), AU (Audit and Accountability),
  CM (Configuration Management), IA (Identification and Authentication),
  IR (Incident Response), SC (System and Communications Protection).

Families NOT in the knowledge base (gaps):
  SA, SR, PT, AT, CA, CP, MA, MP, PE, PL, PM, PS, RA, SI.

Generate a JSON array of exactly 15 test cases.
Each object must have exactly these keys:
  "query"     : a realistic natural-language compliance question
  "expected"  : one of "FULLY_SUPPORTED" | "PARTIALLY_SUPPORTED" | "NO_INFORMATION"
  "rationale" : one sentence explaining why you assigned that label

Distribution REQUIRED:
  5 × FULLY_SUPPORTED    (clearly answered by AC/AU/CM/IA/IR/SC alone)
  5 × PARTIALLY_SUPPORTED (topic spans both included AND excluded families)
  5 × NO_INFORMATION     (only covered by excluded families)

Return ONLY the JSON array. No markdown fences, no explanation, no preamble.
"""

def generate_test_cases() -> list[TestCase]:
    print("[INFO] Generating test cases with Ollama …")
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system",  "content": TESTGEN_SYSTEM},
            {"role": "user",    "content": "Generate the 15 test cases now."},
        ],
    )
    raw = response["message"]["content"].strip()

    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                raw = part
                break

    data  = json.loads(raw)
    cases = [TestCase(**item) for item in data]
    print(f"[INFO] Generated {len(cases)} test cases.")
    return cases

def run_evaluation(
    pipeline:   RAGPipeline,
    test_cases: list[TestCase],
) -> list[EvalResult]:
    results: list[EvalResult] = []
    for i, tc in enumerate(test_cases, 1):
        print(f"[{i:02d}/{len(test_cases)}] {tc.query[:60]} …")
        resp  = pipeline.query(tc.query)
        match = resp.support_level.value == tc.expected
        results.append(EvalResult(
            query=          tc.query,
            expected=       tc.expected,
            actual=         resp.support_level.value,
            top_score=      resp.top_score,
            match=          match,
            answer_preview= resp.answer[:150],
        ))
    return results

def print_report(cases: list[TestCase], results: list[EvalResult]) -> None:
    total   = len(results)
    correct = sum(r.match for r in results)

    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)
    print(f"Total  : {total}")
    print(f"Correct: {correct}  ({correct/total:.0%})")
    print()

    for label in ("FULLY_SUPPORTED", "PARTIALLY_SUPPORTED", "NO_INFORMATION"):
        subset = [r for r in results if r.expected == label]
        if subset:
            hits = sum(r.match for r in subset)
            print(f"  {label:25s}  {hits}/{len(subset)}")

    mismatches = [r for r in results if not r.match]
    if mismatches:
        print(f"\nMismatches ({len(mismatches)}):")
        for r in mismatches:
            rat = next((c.rationale for c in cases if c.query == r.query), "")
            print(f"\n  Query     : {r.query}")
            print(f"  Expected  : {r.expected}")
            print(f"  Actual    : {r.actual}  (top_score={r.top_score:.3f})")
            print(f"  Rationale : {rat}")
            print(f"  Preview   : {r.answer_preview} …")
    else:
        print("\nAll predictions matched ✓")

    print("=" * 70)

def save_results(
    cases:   list[TestCase],
    results: list[EvalResult],
    out_dir: str = RESULTS_DIR,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    cases_path   = os.path.join(out_dir, f"test_cases_{ts}.json")
    results_path = os.path.join(out_dir, f"eval_results_{ts}.json")

    with open(cases_path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in cases], f, indent=2, ensure_ascii=False)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Saved → {cases_path}")
    print(f"[INFO] Saved → {results_path}")

def main() -> None:
    pipeline   = RAGPipeline()
    cases      = generate_test_cases()
    results    = run_evaluation(pipeline, cases)
    print_report(cases, results)
    save_results(cases, results)


if __name__ == "__main__":
    main()
