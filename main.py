import sys
from generator import RAGPipeline, SupportLevel

def run_cli():
    print("\n[INFO] Initialising NIST Assistant...")
    try:
        pipeline = RAGPipeline()
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
            print(f"\n[Support Level: {resp.support_level}]")
            print(f"Answer: {resp.answer}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[ERROR] {e}")

if __name__ == "__main__":
    run_cli()