# Image Aesthetics Model

A dedicated project for designing, training, and benchmarking a **learned multi-signal aesthetic scoring model** for the [Moments Generator](https://github.com/rushivyas/moments-generator) application.

## Why This Project Exists

The Moments Generator application ingests a user's personal photo and video library (up to 20GB), understands the visual content using AI embeddings, and automatically curates highlight reels matching a natural-language prompt like _"show me the best sunset moments from my trip to Iceland."_

At the heart of this pipeline is a **scoring system** that decides which frames and scenes are "the best." Today, this scoring is done with a **hand-tuned weighted sum** of 10 different quality signals. This works, but it has a hard ceiling — the weights are static, the interactions between signals are assumed to be linear, and they can't adapt to different types of content.

This project exists to **replace that heuristic with a learned model** that can discover the optimal way to fuse those signals into a single, accurate aesthetic quality score.

## Project Scope

> **This project is built explicitly and exclusively for the Moments Generator application. It is not a general-purpose IQA library.**

Everything in this repo — the data pipelines, the model architectures, the training scripts, and the evaluation benchmarks — is designed to produce a model that will be deployed back into the Moments Generator's ingestion and ranking pipeline, running locally on Apple Silicon hardware.

## Documentation

| Document | Description |
|:---|:---|
| [Problem Statement](docs/problem-statement.md) | What the Moments Generator is, how it works, what scoring problem we're solving, and why |
| [Input Signals](docs/signals.md) | Detailed description of every signal available, how each is computed, what it captures, and its characteristics |
| [Architecture Alternatives](docs/architecture-alternatives.md) | Full brainstorming of 5 architectural approaches: MLP, Mixture of Experts, Contextual Bandit, Transformer, and Hybrid Supervised+RL |
| [SigLIP 2 Aesthetic Training Plan](docs/siglip2-laion-training-plan.md) | Architecture, cache contract, training protocol, evaluation gates, portability, and implementation phases for the first dependency model |
| [SigLIP 2 Resolution Benchmark Plan](docs/siglip2-resolution-benchmark-plan.md) | Luna-ready plan for benchmarking Base 224 against matching Base 384 on the production M5 Pro pipeline |
| [AVA SigLIP 2 Training Implementation Plan](docs/ava-siglip2-training-implementation-plan.md) | Data acquisition, external storage, leakage-resistant splits, pre-embedding, architecture search, training, evaluation, and export plan for Base 224 |
| [Final Multi-Signal Scorer Plan](docs/final-fusion-model-plan.md) | Independent dataset portfolio, leakage controls, critical architecture review, selected fusion design, training protocol, and release gates |
| [AGENTS.md](AGENTS.md) | Rules and context for AI agents working on this repo |

## Hardware Target

All training and inference must run on:
- **Apple M5 Pro** with **24 GB unified memory**
- PyTorch with MPS (Metal Performance Shaders) backend
- No cloud GPUs, no CUDA dependencies

## Current Status

🧪 **Phase: Fusion planning and data acquisition** — The SigLIP2 A1 head is finalized
and exported after one locked-test evaluation. The independent multi-dataset fusion
pipeline is now specified for implementation.
