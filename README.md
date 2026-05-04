# Evaluation Suite - NLP Assignment 5

This folder contains the standalone evaluation suite for the Daraz AI Chatbot. 
It is designed to be run against the active backend API.

## Setup
1. Ensure the backend is running (`uvicorn main:app ...` or via Docker).
2. Install the requirements for the eval suite:
   ```bash
   pip install aiohttp websockets psutil
   ```

## Running Evaluations
To run the full suite (Correctness + Performance + Throughput) and generate a report:
```bash
python run_evals.py
```

## Contents
- `data.py`: Contains the 30+ RAG QA pairs (with Document IDs), 10 multi-turn dialogues, and tool test cases.
- `run_evals.py`: The main automation script. It measures TTFT, E2E latency, and accuracy metrics.
- `eval_report.md`: (Generated after running) The final Markdown report for submission.

## Metrics
- **RAG Precision**: Calculated by comparing LLM responses to annotated ground truth answers using keyword heuristics.
- **Latency**: Measured over WebSocket to capture accurate **Time to First Token (TTFT)**.
- **Throughput**: Calculated by simulating concurrent users and measuring turns per second.
