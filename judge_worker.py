"""
judge_worker.py
---------------
Long-running judge process that runs INSIDE the Docker backend container.
Loaded once via: docker exec -i hybrid-rag-agent-backend-1 python3 -

Protocol (stdin/stdout, newline-delimited JSON):
  IN  -> {"question": "...", "expected": "...", "actual": "..."}
  OUT <- {"passed": true, "verdict": "YES"}
"""
import sys, json
from llama_cpp import Llama

MODEL_PATH = "/models/qwen2.5-3b-instruct-q4_k_m.gguf"

JUDGE_SYSTEM = (
    "You are a strict automated evaluator. "
    "You receive a question, a reference answer, and an AI's actual response. "
    "Decide if the AI's response correctly addresses the question.But don't be too strict. If reference answer and actual response have same meaning then output YES. "
    "Reply with exactly ONE word: YES or NO. Nothing else."
)

sys.stderr.write("[judge_worker] Loading model...\n")
sys.stderr.flush()

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=1024,
    n_threads=4,
    n_gpu_layers=0,
    verbose=False,
)

sys.stderr.write("[judge_worker] Ready\n")
sys.stderr.flush()
# Signal ready to the parent process
sys.stdout.write(json.dumps({"ready": True}) + "\n")
sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        question = req["question"]
        expected = req["expected"]
        actual   = req["actual"]

        prompt = (
            f"Question: {question}\n"
            f"Reference answer: {expected}\n"
            f"AI's response: {actual}\n\n"
            "Did the AI correctly answer the question? Reply YES or NO."
        )

        result = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=5,
            temperature=0.0,
        )
        verdict = result["choices"][0]["message"]["content"].strip().upper()
        passed  = verdict.startswith("YES")

        sys.stdout.write(json.dumps({"passed": passed, "verdict": verdict}) + "\n")
        sys.stdout.flush()
    except Exception as e:
        sys.stdout.write(json.dumps({"passed": False, "verdict": "ERROR", "error": str(e)}) + "\n")
        sys.stdout.flush()
