# Hybrid RAG Agent - Evaluation Suite (Assignment 5)

This repository contains the automated evaluation suite for the Daraz AI Shopping Assistant.

## Demo Video
[Watch the Evaluation Demo (Drive)](https://drive.google.com/file/d/1t_P_NaEnnSjNfQsGQMz7eXefqdP4YjAY/view?usp=sharing)

*(Couldn't upload on YouTube due to the video length, so the Drive link is the submission copy.)*

## Deliverables
- `report.md`: formal evaluation report with tables, analysis, and plots.
- `eval_results.json`: machine-readable results produced by the evaluation suite.
- `plots/`: charts generated from the saved results snapshot.

## Setup
1. Make sure Docker is running. The evaluator launches `judge_worker.py` inside the backend container, so Docker Desktop/Engine must be available.
2. Install the Python packages used by the harness:
   ```powershell
   python -m pip install aiohttp websockets psutil matplotlib
   ```
3. Start the backend API:
   ```powershell
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

## Run the Evaluation Suite
From the repository root:

```powershell
python run_evals.py
```

When the run completes, open the live dashboard at `http://localhost:8765/`.

## Metrics
The suite computes the following metrics:

- **RAG Precision** = `rag_passed / rag_total`
- **Tool Accuracy** = `tool_passed / tool_total`
- **TTFT** (Time to First Token) = time from sending the websocket prompt to receiving the first token
- **ITL** (Inter-Token Latency) = `(E2E - TTFT) / max(1, token_count - 1)`
- **E2E Latency** = total time from sending the prompt until the `done` event
- **95% CI** = `1.96 * std_dev / sqrt(n)` for the latency samples in each scenario
- **Throughput** = completed turns per second during the fixed concurrency sweep

`DockerJudge` runs `judge_worker.py` inside the backend container with a neutral system prompt and compares the actual answer against the expected answer for RAG and tool cases.

## How to Interpret Results
- **RAG Precision**: above 75% is strong for this domain; 50-75% is usable but still noisy; below 50% needs retrieval or prompt work.
- **Tool Accuracy**: above 90% is strong; below 70% usually means tool routing or instrumentation is missing.
- **Latency**: lower is better. For an interactive assistant, TTFT under 5s is good, 5-10s is acceptable, and above 10s is slow.
- **Throughput**: higher is better, but it should be interpreted with latency because fast throughput with slow first-token time is still poor UX.

## Assumptions and Limitations
- The saved report reflects one evaluation snapshot, not a repeated benchmark campaign.
- RAG scoring uses a heuristic overlap check plus an LLM-judge fallback, so semantically correct paraphrases can still be marked false.
- Tool accuracy depends on the backend emitting the correct tool markers over the websocket stream.
- The latency numbers were measured on the available local hardware and include model inference overhead.
- The current plots are derived from the saved `eval_results.json` snapshot; a concurrency-vs-latency curve would require a separate sweep across multiple concurrency levels.

## Project Files
- `data.py`: annotated ground truth for RAG, tool, and latency scenarios.
- `run_evals.py`: evaluation harness and dashboard server.
- `report.md`: formal report for submission.
- `plots/`: generated charts referenced by the report.
