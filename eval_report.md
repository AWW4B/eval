# NLP Assignment 5 - Evaluation Report
Generated on: Mon May  4 20:33:00 2026

## 1. System Configuration
- **OS**: Windows
- **CPU**: AMD64 Family 25 Model 124 Stepping 0, AuthenticAMD
- **RAM**: 15.23 GB
- **Python**: 3.14.3

## 2. Correctness Metrics
| Metric | Value |
|--------|-------|
| RAG Precision@1 | 51.61% |
| Tool Invocation Accuracy | 100.00% |
| Policy Adherence | 100% |

## 3. Performance Metrics (Single Turn)
| Scenario | Mean E2E (ms) | 95% CI | Median (ms) | P90 (ms) | Avg TTFT (ms) |
|----------|---------------|--------|-------------|----------|---------------|
| simple | 2353.0 | ±23.5 | 2344.8 | 2396.2 | 2104.6 |
| rag | 2334.5 | ±11.3 | 2331.5 | 2349.9 | 2094.7 |
| tool | 2321.0 | ±20.9 | 2326.6 | 2343.4 | 2085.4 |
| mixed | 2323.9 | ±17.9 | 2332.3 | 2342.4 | 2089.9 |

## 4. Throughput
- **Sustainable Concurrency**: 5 users
- **Turns per Second**: 13.59
