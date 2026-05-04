# NLP Assignment 5 - Evaluation Report
Generated on: Tue May  5 01:49:20 2026

## 1. System Configuration
- **OS**: Linux
- **CPU**: 
- **RAM**: 14.85 GB
- **Python**: 3.14.4
- **Judge**: Qwen 2.5-3B-Instruct (Docker, neutral system prompt)

## 2. Correctness Metrics
| Metric | Value | Detail |
|--------|-------|--------|
| RAG Precision@1 | 54.84% | 17/31 |
| Tool Accuracy   | 62.50% | 5/8 |

## 3. Latency (Single Turn, 5 trials each)
| Scenario | Mean E2E (ms) | 95% CI | Median (ms) | P90 (ms) | Avg TTFT (ms) |
|----------|---------------|--------|-------------|----------|---------------|
| simple | 18516.6 | ±935.4 | 18298.5 | 20368.0 | 16260.2 |
| rag | 22530.7 | ±3243.6 | 21446.9 | 28594.3 | 18091.0 |
| tool | 23950.6 | ±1510.6 | 23117.0 | 26439.9 | 21775.3 |
| mixed | 28278.9 | ±3014.5 | 26708.6 | 33530.5 | 23134.2 |

## 4. Throughput
- **Turns/sec**: 0.05
- **Duration**: 304.9s