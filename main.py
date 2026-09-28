import sys
import argparse
from pathlib import Path
from generator import RAGPipeline, OLLAMA_MODEL
from retriever import DEFAULT_INDEX_DIR

def run_cli(index_dir=DEFAULT_INDEX_DIR, model=OLLAMA_MODEL):
    print("\n[INFO] Initialising NIST Assistant...")
    try:
        pipeline = RAGPipeline(model=model, index_dir=index_dir)
        print("[INFO] Ready. Type 'exit' to quit.\n")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            if not user_input:
                continue

            resp = pipeline.query(user_input)
            print(f"\n[Retrieval similarity: {resp.retrieval_signal.value}]")
            print(f"[Answer evidence: {resp.evidence_status.value}]")
            print(f"Answer: {resp.answer}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[ERROR] {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask arbitrary questions using a chosen NIST index.")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--model", default=OLLAMA_MODEL)
    args = parser.parse_args()
    run_cli(index_dir=args.index_dir, model=args.model)
