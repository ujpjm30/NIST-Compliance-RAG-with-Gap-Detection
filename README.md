# NIST 800-53 RAG Pipeline with Gap Detection

**Jimin Park**

---

## Quick Start (< 5 minutes)

### 1. Install dependencies

```bash
pip install pandas sentence-transformers faiss-cpu numpy ollama
```

### 2. Pull the local LLM

```bash
ollama pull llama3
# Ensure Ollama is running in the background before execution
```

### 3. Build the index

The system uses a curated dataset of 30 controls to test gap detection.

```bash
python ingestion.py
```

### 4. Run the assistant

```bash
python main.py
```

### 5. Run automated evaluation

```bash
python evaluate.py
```

---

## Project Structure

```
.
├── data/
│   └── nist_corpus.csv              # Curated NIST 800-53 Rev.5 subset (30 controls)
├── faiss_index/
│   ├── index.bin                    # FAISS vector index
│   └── metadata.pkl                 # Chunk metadata
├── eval_results/                    # Raw evaluation output (JSON)
├── chat_logs/
│   └── CHAT_LOGS.md                 # AI collaboration logs (Gemini)
├── ingestion.py                     # CSV → chunks → FAISS index
├── retriever.py                     # Semantic search with similarity scores
├── generator.py                     # Gap detection + Ollama generation
├── main.py                          # Interactive CLI
├── evaluate.py                      # Automated evaluation suite
└── README.md
```

---

## Design Decisions

### LLM & Embeddings

- **LLM**: Ollama + `llama3` — runs entirely local, ensuring data privacy and zero API costs.
- **Embeddings**: `all-MiniLM-L6-v2` via `sentence-transformers` (384-dim), optimized for CPU inference.
- **Architecture**: Intentionally avoided heavy frameworks like LangChain to keep the dependency footprint minimal and ensure full reproducibility.

### Corpus Construction (Intentional Gaps)

To evaluate the system's ability to "know what it doesn't know," I selected controls from **6 families** (configured in `ingestion.py`):

| Family | Max Selected | Description |
|--------|-------------|-------------|
| AC | 6 | Access Control |
| AU | 5 | Audit and Accountability |
| CM | 5 | Configuration Management |
| IA | 5 | Identification and Authentication |
| IR | 4 | Incident Response |
| SC | 5 | System and Communications Protection |

The CSV corpus contains 30 controls (5 per family). Since `ingestion.py` uses `head(count)` per family, the actual indexed set is **29 chunks** (AC caps at 5 due to CSV availability, IR takes only 4).

Families **intentionally excluded** to simulate real-world knowledge gaps: SA, SR, PT, AT, CA, CP, MA, MP, PE, PL, PM, PS, RA, SI.

Queries about missing families (e.g., Physical Security — PE, or Security Assessment — CA) correctly trigger a `NO_INFORMATION` status.

### FAISS Indexing

- **Index**: `IndexFlatIP` (Inner Product) with L2-normalized embeddings.
- **Rationale**: This setup yields exact cosine similarity. Since the corpus is small (29 chunks), a flat index provides perfect recall without the overhead of approximate methods like HNSW.

### Gap Detection Logic

Score-based classification using cosine similarity thresholds:

| Condition | Label |
|-----------|-------|
| top score < 0.30 | `NO_INFORMATION` |
| top score ≥ 0.30 AND fewer than 2 docs ≥ 0.55 | `PARTIALLY_SUPPORTED` |
| ≥ 2 docs with score ≥ 0.55 | `FULLY_SUPPORTED` |

Requiring **two** high-confidence matches significantly reduces false positives where a single document might have accidental semantic overlap with the query.

### Citations & Prompting

- The system prompt enforces strict `[Control_ID]` inline citations for every factual claim.
- Zero-temperature generation ensures deterministic, grounded outputs.
- If no supporting evidence is found, the model is instructed to refuse the answer rather than hallucinate control IDs.

---

## Evaluation Results

Run `python evaluate.py` to reproduce the automated report in `./eval_results/`.

15 test cases across the three coverage tiers:

| Expected Label | Correct | Notes |
|----------------|---------|-------|
| FULLY_SUPPORTED | 5/5 | High reliability for AC/IA/AU families. |
| PARTIALLY_SUPPORTED | 4/5 | One cross-family query misclassified. |
| NO_INFORMATION | 4/5 | High-scoring overlap in SC family triggered a false partial. |

**Estimated Accuracy: ~87%**

### Mismatch Analysis

- **Semantic Overlap**: A query about "secure software development" (SA family, excluded) partially mapped to SC-28 (Protection of Information at Rest), resulting in `PARTIALLY_SUPPORTED` instead of `NO_INFORMATION`. This is a known limitation of purely distance-based gap detection — semantically adjacent but topically distinct controls can produce misleading scores.
- **Ambiguity**: Queries spanning both included and excluded families occasionally lean toward `FULLY_SUPPORTED` if the included family's controls are highly relevant to the query's surface-level phrasing.

---

## What I'd Improve with More Time

- **Re-ranking**: Implement a cross-encoder re-ranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) to refine precision after the initial bi-encoder retrieval.
- **Structured Verification**: Post-process LLM output to ensure every cited `[Control_ID]` actually exists in the retrieved context window, catching hallucinated citations.
- **Recursive Chunking**: Improve handling of hierarchical NIST sub-controls (e.g., AC-2(1)) during ingestion to preserve parent-child relationships.
- **Family-Aware Gap Detection**: Add a family-check layer that verifies whether the retrieved chunks actually span the required families for complex, cross-domain queries.
- **NLI-Based Classification**: Replace or augment threshold-based gap detection with a Natural Language Inference model for more robust "PARTIALLY_SUPPORTED" classification.

---

## AI Tool Usage

In accordance with project guidelines, this project was developed with assistance from **Gemini (Google)**. Technical discussions, problem definitions, and troubleshooting logs are documented in `chat_logs/CHAT_LOGS.md`.
