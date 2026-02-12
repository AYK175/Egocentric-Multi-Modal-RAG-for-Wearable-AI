# Egocentric Multi-Modal RAG for Wearable AI
**Optimizing Llama 3.2 11B Vision for the CRAG-MM Benchmark**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This repository implements a high-performance **Multi-Modal Retrieval-Augmented Generation (RAG)** system designed for egocentric (first-person) wearable devices. Built on the **Llama 3.2 11B Vision** model, our system addresses the critical challenge of wearable AI: processing noisy visual data while accurately retrieving external factual knowledge.

### Key Achievements:
* **Performance Boost:** Increased benchmark accuracy from **63.0% to 67.0%** through LoRA fine-tuning.
* **The Quantization Paradox:** Identified and analyzed the trade-off where 4-bit quantization reduces memory footprint but increases inference latency due to dequantization overhead.
* **Sub-Query Decomposition:** Implemented a dual-model architecture using a Llama 8B agent to break complex queries into searchable sub-queries for superior retrieval.

---

## System Architecture
The pipeline follows a modular **Search-Filter-Generate** flow:
1.  **Vision Encoder:** Llama 3.2 11B Vision identifies objects in the egocentric frame.
2.  **Sub-Query Agent:** A secondary Llama 3.1 8B model decomposes user questions into atomic search queries.
3.  **Unified Search Pipeline:** Retrieves web context while applying a **strict top-k filter** to mitigate noise.
4.  **Context-Aware Generation:** The MLLM synthesizes visual data and retrieved text into a factual response.

## 🔍 Failure Analysis & Insights
During development, we identified two critical bottlenecks that informed our final architecture:

### A. The Parametric Knowledge Deficit
Even after fine-tuning the vision encoder to recognize complex objects (e.g., a 1969 Chevrolet Camaro), the language backbone often failed to recall specific technical facts (e.g., engine displacement or historical significance).
* **Solution:** We offloaded factual recall to our **Sub-Query RAG Pipeline**, using the model primarily for visual interpretation rather than memorization.

### B. The Quantization Paradox
We observed that while 4-bit quantization (via BitsAndBytes) reduced the memory footprint by ~70%, it introduced a "dequantization overhead" that slightly increased per-token latency on certain GPU architectures.
* **Engineering Decision:** We opted for **Unsloth-optimized kernels** to regain this lost speed, achieving a 2x inference boost over standard Hugging Face implementations.

---

## Project Structure
```text
.
├── configs/            # LoRA & RAG hyperparameters
├── notebooks/          # Step-by-step research evolution
├── src/                # Production-ready modular code
│   ├── inference.py    # Optimized 4-bit MLLM inference
│   ├── rag_engine.py   # RAG pipeline & strict context filtering
│   ├── search_query.py # Llama 8B sub-query generation logic
│   └── eval_metrics.py # Automated regex scoring & analytics
├── Research_Paper.pdf  # Full academic methodology and findings
└── requirements.txt    # Optimized dependency list (Unsloth, PEFT, etc.)

