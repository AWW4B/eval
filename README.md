# Hybrid RAG Agent - Evaluation Suite (Assignment 5)

This repository contains the automated evaluation suite for the Daraz AI Shopping Assistant.

## 📺 Demo Video
[Watch the Evaluation Demo (Loom)](https://vimeo.com/placeholder)  
*(Replace with your recorded video link)*

## 🚀 How to Run
1. **Start the Chatbot Backend**:
   ```bash
   cd backend/src
   uvicorn main:app --port 8000
   ```
2. **Execute Evaluations**:
   ```bash
   cd evalsetc
   pip install -r requirements.txt
   python run_evals.py
   ```
3. **View Results**:
   The suite will generate `eval_report.md` (raw) and you can find the formal analysis in `report.md`.

## 📊 Metrics Definitions
- **TTFT (Time to First Token)**: Measured from the moment the JSON is sent over WebSocket until the first token message is received.
- **RAG Precision**: Calculated as the percentage of queries where the retrieved context contains keywords from the annotated ground truth.
- **Faithfulness**: A heuristic check verifying that the LLM response contains specific factual markers from the retrieved document.
- **95% Confidence Interval**: Calculated using the standard deviation of 30 trials: `1.96 * (std_dev / sqrt(n))`.

## 📂 Folder Structure
- `/data.py`: Annotated ground truth for 30+ RAG queries and 10+ dialogues.
- `/run_evals.py`: The single-command test harness.
- `/report.md`: Formal analysis and performance curves.

## 🛠 Dependencies
- `aiohttp`: Async HTTP requests.
- `websockets`: Real-time token latency measurement.
- `psutil`: Hardware configuration detection.
- `numpy`: Statistical calculations.
