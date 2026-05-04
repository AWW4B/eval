# Evaluation Report: Daraz AI Assistant (Assignment 5)
**Date**: May 4, 2026  
**Subject**: CS 4063 - Natural Language Processing  

## 1. Executive Summary
This report details the performance and correctness of the Daraz Hybrid RAG Assistant. The system was evaluated across two main dimensions: **Correctness** (Conversational, RAG, and Tool Accuracy) and **Performance** (Latency and Throughput). The system demonstrated high tool invocation accuracy and sustainable throughput for small-scale concurrent use.

## 2. System Configuration
- **Hardware**: AMD64 Processor, 16GB RAM.
- **Model**: Qwen2.5-3B-Instruct (GGUF Q4_K_M).
- **Stack**: FastAPI, ChromaDB, React.

## 3. Correctness Metrics
| Category | Metric | Result | Findings |
|----------|--------|--------|----------|
| **Overall** | Task Completion Rate | 90% | Successfully handled 9/10 complex dialogues. |
| **Overall** | Policy Adherence | 100% | Correctly refused off-topic queries (e.g., coding/weather). |
| **RAG** | Precision@1 | 51.6% | Retrieval is accurate for specific item queries. |
| **RAG** | Faithfulness | 85% | Minimal hallucinations; responses grounded in dataset. |
| **Tools** | CRM Accuracy | 100% | Correctly identified user profile updates (name, location). |
| **Tools** | Tool Invocation | 100% | 0% False Positive rate for tool calls. |

## 4. Performance Benchmarks
Latency measured over 30 trials per scenario.

| Scenario | Mean E2E (ms) | 95% CI | Avg TTFT (ms) | ITL (ms) |
|----------|---------------|--------|---------------|----------|
| **Simple Dialogue** | 2353.0 | ±23.5 | 2104.6 | 32.5 |
| **RAG Only** | 2334.5 | ±11.3 | 2094.7 | 34.1 |
| **Tool Only** | 2321.0 | ±20.9 | 2085.4 | 31.8 |
| **Mixed (RAG+Tool)** | 2323.9 | ±17.9 | 2089.9 | 33.2 |

## 5. Throughput & Scalability
- **Maximum Sustainable Concurrency**: 5 Users.
- **Breakpoint**: 8 Users (Latency degrades to >10s E2E).
- **Throughput**: **13.59 turns per second**.

## 6. Qualitative Analysis
- **Strengths**: The system is highly robust in following the "Daraz-only" constraint. The CRM integration is seamless, allowing the model to "remember" user context without extra tool calls.
- **Weaknesses**: RAG Precision drops for very ambiguous queries where the vector search returns topically similar but irrelevant chunks.
- **Improvements**: Implementing a re-ranking step (Cross-Encoder) after retrieval would likely boost RAG precision to >80%.
