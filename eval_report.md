# NLP Assignment 5 - Evaluation Report
Generated on: Tue May  5 00:31:10 2026

## 1. System Configuration
- **OS**: Linux
- **CPU**: 
- **RAM**: 14.85 GB
- **Python**: 3.14.4
- **Judge**: Qwen 2.5-3B-Instruct (Docker, neutral system prompt)

## 2. Correctness Metrics
| Metric | Value | Detail |
|--------|-------|--------|
| RAG Precision@1 | 0.00% | 0/31 |
| Tool Accuracy   | 0.00% | 0/8 |

## 3. Latency (Single Turn, 5 trials each)
| Scenario | Mean E2E (ms) | 95% CI | Median (ms) | P90 (ms) | Avg TTFT (ms) |
|----------|---------------|--------|-------------|----------|---------------|
| simple | 18169.0 | ±2505.0 | 18406.2 | 20807.4 | 15995.1 |
| rag | 20161.1 | ±2407.6 | 20055.5 | 24164.7 | 15550.9 |
| tool | 25160.1 | ±2011.2 | 24101.3 | 29229.8 | 23316.5 |
| mixed | 32420.9 | ±8003.8 | 29518.5 | 47852.2 | 25790.2 |

## 4. Throughput
- **Turns/sec**: 0.05
- **Duration**: 282.8s