# NIST 800-53 RAG Pipeline with Gap Detection

**Jimin Park**

## Run locally

Use the project's Python environment and start the Ollama application with `llama3` installed. Model files must be downloaded before offline use.

```bash
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3
python ingestion.py
python main.py
```

On this Mac, if `ollama` launches the GUI instead of the command-line tool, use `/usr/local/bin/ollama` for CLI commands. The ingestion and retrieval code limit native worker threads on macOS because the installed native libraries previously crashed during encoding.

`main.py` accepts arbitrary customer questions. Evaluation questions do not restrict what customers can ask.

## Corpus and chunking

The input is `data/NIST_SP-800-53_rev5_catalog_load.csv`, the local Rev 5 snapshot: 1,189 entries across 20 families (322 base entries and 867 enhancements). This includes 180 withdrawn entries. It is a local snapshot, not a claim to track subsequent NIST updates.

The default `full` strategy stores one entry per chunk, including its ID, title, control text, and Discussion. It writes `faiss_index_full_baseline/`. The embedding model has a configured 512-token input limit, so long full-control chunks may still be truncated by the tokenizer. New builds report and record the number of affected chunks in `manifest.json`.

The original prototype selected a small subset to simulate corpus gaps and retained only the first 500 characters of each control's body. That was truncation: the remainder was not stored in later chunks. The current optional `chars` strategy preserves the entire assembled text across multiple chunks:

```bash
python ingestion.py --strategy chars --chunk-size 500 --overlap 50
```

Each chunk is at most 500 characters, including a repeated control ID and title. Adjacent body segments overlap by 50 characters, with paragraph, sentence, or word boundaries preferred where possible. Metadata records source offsets and the original control ID. This configuration produces 3,431 chunks from the same 1,189 entries and writes `faiss_index_chars_500_overlap_50/`.

Every piece also inherits `is_withdrawn` and the original withdrawal notice. Normal retrieval excludes withdrawn entries before selecting the top five. Rebuild old indexes using their original strategy if the reader reports missing withdrawal metadata. Historical entries remain stored; explicitly including them requires the retriever's `include_withdrawn=True` API option, and their status is included in the answer context.

500 characters is an experiment setting, not an established optimum, and characters are not tokens. Short chunks may improve access to later requirements but lose surrounding context. Full and character indexes can be compared using the same saved questions. Parent and enhancement entries are not merged. NIST cross-references are retained as metadata; they do not automatically become correct-answer labels.

## Generate new questions with Llama

Question generation is a separate command. It samples active controls from multiple families and cycles through direct questions, paraphrases, scenarios, questions spanning two controls, and questions requiring customer implementation evidence.

```bash
python generate_questions.py --count 15 --seed 42 --output eval_questions/customer_v1.json
```

Use `--language ko` for Korean questions. Omit `--seed` for new random source selection. Existing output files are protected from overwriting. Each completed question is saved, so an interrupted run retains its completed questions. The file records the selected source excerpts, model, seed, and corpus hash. A seed fixes source selection; model output can still vary across versions.

Generated answers and source IDs are drafts requiring review. Each case starts with `expected: null` and `review_status: "unreviewed"`. Source ID validation catches references outside the supplied excerpts; it does not prove that an answer is correct or that a question represents actual customer traffic. Review questions for answerability, coverage, and realistic wording before treating them as a benchmark.

## Evaluate a saved set

Run both strategies against the same saved questions:

```bash
python evaluate.py --cases eval_questions/customer_v1.json --index-dir faiss_index_full_baseline
python evaluate.py --cases eval_questions/customer_v1.json --index-dir faiss_index_chars_500_overlap_50
```

Without `--cases`, evaluation uses the original saved 15 questions in `eval_results/test_cases_20260405_225548.json`. Those old labels were associated with the earlier corpus and require review for the current corpus. Keeping a saved set makes changes comparable; generating additional sets expands coverage.

Reports in `eval_results/` contain answers, retrieved chunk IDs and text, withdrawal status, question intent, retrieval signals, evidence statuses, citation checks, timings, and hashes identifying the cases, index, and code. They do not calculate accuracy by default. To compare manually reviewed routing outcomes, set each case's `review_status` to `"reviewed"` and add `expected_evidence_status` using one of the evidence statuses below, then use:

```bash
python evaluate.py --cases eval_questions/customer_v1.json --score-reviewed
```

This reports evidence-status agreement only, not answer factuality or retrieval recall. Original `expected` support labels are retained for provenance but cannot be used as evidence-status labels. The earlier approximately 87% result is not a validated score for this full corpus or either current chunking strategy. Comparisons should distinguish corpus expansion, removal of truncation, and the subsequent chunking change.

To use character chunks interactively:

```bash
python main.py --index-dir faiss_index_chars_500_overlap_50
```

## Retrieval signals and answer evidence

Embeddings use `all-MiniLM-L6-v2` on CPU. L2-normalized vectors are searched with FAISS `IndexFlatIP`, providing exact cosine nearest neighbors within the stored vectors. This does not guarantee that the nearest text answers the question. The five highest-scoring active chunks are passed to generation when a requirements answer is requested.

The current heuristic assigns:

- `LOW_SIMILARITY` when the best similarity is below 0.30.
- `HIGH_SIMILARITY` when at least two distinct control IDs score at least 0.55.
- `MODERATE_SIMILARITY` otherwise.

Multiple chunks from one control count as one control for this decision. Thresholds have not been recalibrated for the expanded corpus or character chunks. These labels describe a retrieval heuristic, not verified answerability or an organization's compliance.

Evidence status is reported separately:

- `ORGANIZATION_EVIDENCE_REQUIRED`: the question asks about actual organizational implementation. The code returns a fixed limitation statement, because this corpus contains no organization-specific evidence. It does not ask the answer model to infer compliance from NIST requirements.
- `CLARIFICATION_REQUIRED`: the question's intent could not be classified safely.
- `INSUFFICIENT_CONTEXT`: a requirements question has low retrieval similarity.
- `INVALID_CITATIONS`: a generated answer has no control citations or cites IDs outside the retrieved active controls; it is replaced with a review message.
- `NEEDS_REVIEW`: a generated requirements answer passed the citation membership check. Factual support for its claims is still unverified.

Common explicit organization-status questions in English and Korean are routed by rules. A separate local Llama classification handles other wording; invalid classification output requests clarification. Intent classification can still make mistakes and needs broader customer testing. Neither evaluation labels nor reference answer drafts are passed to the pipeline.

Citation checking verifies exact source ID membership, not entailment or that every statement is cited. Requirements answers use only catalog context and must not assert actual organizational compliance. Temperature zero does not guarantee factuality or reproducibility. Full controls remain the default; the 500-character strategy remains an experiment.

## Files

- `ingestion.py`: corpus parsing, full/character chunks, embeddings, index manifests.
- `retriever.py`: chosen-index loading and semantic retrieval.
- `generator.py`: question intent, evidence guards, citation checks, and Ollama answers.
- `main.py`: interactive assistant with `--index-dir` and `--model` options.
- `generate_questions.py`: separate Llama question generation and checkpointing.
- `evaluate.py`: saved-question evaluation and optional reviewed-label scoring.
- `tests/test_workflows.py`: lossless chunk coverage and question validation checks.

## AI assistance

The original prototype used Gemini; earlier discussions are in `chat_logs/CHAT_LOGS.md`. Subsequent corpus, runtime, chunking, and evaluation changes were developed with Codex assistance.
