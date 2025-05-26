# LLM from Scratch: A Complete Learning Roadmap 🚀

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#license) [![GitHub issues](https://img.shields.io/github/issues/username/llm-masterplan.svg)](#) [![Contributors](https://img.shields.io/github/contributors/username/llm-masterplan.svg)](#)

## Overview

An **exhaustive**, phase-driven curriculum to build Large Language Models (LLMs) entirely from first principles. Each phase guides you through theory, implementation, and real-world projects—perfect for educational content, video tutorials, and hands-on articles.

> 🌟 **Why This Roadmap?**
>
> - **Modular & Scalable**: Tackle one phase at a time or customize your learning path.
> - **Project-Focused**: Every phase has clear deliverables and assignments.

---

## 🗂 Table of Contents

1. [Prerequisites & Foundations](#phase-0-prerequisites--foundations)
2. [Data Collection & Preprocessing](#phase-1-data-collection--preprocessing)
3. [Core Transformer Architecture](#phase-2-core-transformer-architecture)
4. [Training Loop & Optimization](#phase-3-training-loop--optimization)
5. [Pre-Training Objectives & Strategies](#phase-4-pre-training-objectives--strategies)
6. [Fine-Tuning & Parameter-Efficient Techniques](#phase-5-fine-tuning--parameter-efficient-techniques)
7. [Model Compression & Efficiency](#phase-6-model-compression--efficiency)
8. [Extended Context & Memory Mechanisms](#phase-7-extended-context--memory-mechanisms)
9. [Sparse & Mixture-of-Experts](#phase-8-sparse--mixture-of-experts)
10. [Multimodal & Cross-Modal Architectures](#phase-9-multimodal--cross-modal-architectures)
11. [Autonomous Agents & Reasoning](#phase-10-autonomous-agents--reasoning)
12. [Safety, Alignment & Evaluation](#phase-11-safety-alignment--evaluation)
13. [Deployment & MLOps](#phase-12-deployment--mlops)
14. [Cutting-Edge Research & Future Directions](#phase-13-cutting-edge-research--future-directions)

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/theunnecessarythings/llm-101.git
cd llm-101

# Install uv if not already installed
# https://docs.astral.sh/uv/getting-started/installation/

uv sync
```

> 🔧 _Ensure CUDA toolkit and drivers are installed for GPU acceleration._

---

## Phase 0: Prerequisites & Foundations

### 0.1 Mathematics & Theory

- [x] **Linear Algebra**: vectors, matrices, eigenvalues/eigenvectors, SVD.
- [x] **Calculus**: derivatives, gradients, Jacobians, Hessians.
- [x] **Probability & Statistics**: distributions, expectation, variance, Bayes’ theorem.
- [x] **Optimization**: gradient descent, Adam, L-BFGS.

### 0.2 Programming & Tooling

- [x] Master Python (data structures, OOP, functional paradigms).
- [x] Git & version control best practices.
- [x] Linux shell & scripting.
- [x] Docker & containerization basics.
- [x] Familiarity with PyTorch and/or TensorFlow low‑level APIs.

---

## Phase 1: Data Collection & Preprocessing

### 1.1 Corpus Assembly

- [ ] Web crawler to harvest text from various domains.
- [ ] Data deduplication (shingling or MinHash).
- [ ] Language detection & filtering.

### 1.2 Cleaning & Normalization

- [ ] Unicode normalization (NFC/NFD).
- [ ] Remove boilerplate (HTML tags, code snippets).
- [ ] Sentence segmentation and truecasing.

### 1.3 Tokenization & Vocabulary

- [ ] Implement **Byte‑Pair Encoding (BPE)** from scratch.
- [ ] Extend to **WordPiece** and **Unigram** algorithms (SentencePiece).
- [ ] Build subword vocab, handle unknown tokens, vocab sizing experiments.

### 1.4 Embedding Initialization

- [ ] Learnable token embeddings.
- [ ] Sinusoidal positional embeddings (analysis of periodicity).
- [ ] Learned positional embeddings; compare performance.

**Deliverable:** Cleaned, tokenized dataset with train/validation splits and embedding initializer module.

---

## Phase 2: Core Transformer Architecture

### 2.1 Scaled Dot‑Product Attention

- [ ] Q/K/V projections.
- [ ] Attention score matrix, scaling factor, masked & unmasked variants.

### 2.2 Multi‑Head Mechanism

- [ ] Head splitting & parallel computation.
- [ ] Concatenate & output projection.

### 2.3 Feed‑Forward Networks

- [ ] Two-layer MLP, activation functions (ReLU, GELU).
- [ ] Parameter initialization strategies (Xavier, Kaiming).

### 2.4 Normalization & Residuals

- [ ] **LayerNorm** implementation (pre‑norm vs post‑norm).
- [ ] Residual connections, understanding gradient flow.

### 2.5 Full Encoder & Decoder Stacks

- [ ] Encoder: stack N layers.
- [ ] Decoder: causal masking, cross‑attention.
- [ ] Model config class: hyperparameters management.

- [ ] **Assignment:** Build a minimal Transformer language model and verify shapes with dummy data.

---

## Phase 3: Training Loop & Optimization

### 3.1 Training Infrastructure

- [ ] Custom `Dataset` and `DataLoader` classes with dynamic batching.
- [ ] Gradient accumulation for large batch emulation.

### 3.2 Optimization Algorithms

- [ ] SGD, Adam, AdamW; implement weight decay correctly.
- [ ] Learning rate schedules: linear warmup, cosine decay.

### 3.3 Precision & Parallelism

- [ ] Mixed precision (FP16/BF16) with PyTorch AMP.
- [ ] Data parallelism (DDP), model parallelism (tensor & pipeline).
- [ ] ZeRO‐style optimizer states offloading.

### 3.4 Monitoring & Logging

- [ ] Training loss, eval metrics, gradient norms.
- [ ] TensorBoard or Weights & Biases integration.

- [ ] **Deliverable:** Scalable training script capable of pretraining a small-scale model on multi‑GPU.

---

## Phase 4: Pre‑Training Objectives & Strategies

### 4.1 Masked Language Modeling (MLM)

- [ ] Random vs whole‑word masking strategies.
- [ ] Span masking (SpanBERT) and n‑gram masking.

### 4.2 Causal Language Modeling (CLM)

- [ ] Left‑to‑right autoregressive loss.
- [ ] Curriculum learning: variable context lengths.

### 4.3 Sequence‑to‑Sequence & Permutation

- [ ] T5 span-corruption objective.
- [ ] XLNet’s permutation language modeling.

### 4.4 Curriculum & Continual Learning

- [ ] Progressive layer unfreezing.
- [ ] Task‑aware sampling, domain‑adaptive pretraining.

- [ ] **Project:** Pretrain BERT, GPT, and T5 variants on small corpus; compare convergence.

---

## Phase 5: Fine‑Tuning & Parameter‑Efficient Techniques

### 5.1 Full Fine‑Tuning

- [ ] End‑to‑end gradient updates.
- [ ] Hyperparameter sweep for learning rates, batch sizes.

### 5.2 Adapters & LoRA

- [ ] Insert adapter modules; experiment with bottleneck sizes.
- [ ] LoRA low‑rank update formulation, implement QLoRA for 4‑bit quantized backends.

### 5.3 Prefix & Prompt Tuning

- [ ] Fixed prompt vectors, virtual tokens.
- [ ] P-tuning v2: deep prefix tuning.

### 5.4 Instruction Tuning & RLHF

- [ ] Curate instruction‑response dataset.
- [ ] Implement PPO for reward‑model‑based alignment.

- [ ] **Deliverable:** Fine-tuned models with adapters and LoRA, benchmarked on GLUE, SQuAD.

---

## Phase 6: Model Compression & Efficiency

- [ ] **Quantization**: Post‑training quantization INT8, GPTQ for FP4.
- [ ] **Pruning**: structured vs unstructured, iterative magnitude pruning.
- [ ] **Distillation**: Teacher-student training, distillation temperature and loss terms.
- [ ] **Sparse Models**: implement simple sparse attention, measure speedups.

- [ ] **Assignment:** Compress a GPT-style model to <1/4 parameters; measure perplexity trade‑off.

---

## Phase 7: Extended Context & Memory Mechanisms

- [ ] **Local/Global Attention**: Longformer & BigBird implementations.
- [ ] **RoPE Rescaling**: Rotary embeddings for >65K tokens.
- [ ] **KV Cache Cascading**: incremental decoding with unbounded context.
- [ ] **Retrieval-Augmented Generation**: build vector store (FAISS/Annoy), integrate RAG.

- [ ] **Project:** Build a chatbot with 100K token context window and external document retrieval.

---

## Phase 8: Sparse & Mixture‑of‑Experts

- [ ] **Expert Layers**: implement routing networks, capacity factors.
- [ ] **Switch & GShard**: single vs multi-token MoE.
- [ ] **Load Balancing Loss**: auxiliary losses for expert utilization.

- [ ] **Deliverable:** Train a toy MoE model; compare training speed and performance to dense baseline.

---

## Phase 9: Multimodal & Cross‑Modal Architectures

- [ ] **Vision–Language**: CLIP pretraining, cross‑attention modules.
- [ ] **Flamingo‑Style**: gated cross‑modal blocks; few‑shot on images.
- [ ] **Audio & Video**: spectrogram encoders, temporal attention.

- [ ] **Project:** Build a multimodal model that generates captions and answers questions about images.

---

## Phase 10: Autonomous Agents & Reasoning

- [ ] **ReAct**: interleaved reasoning and action.
- [ ] **Tree‑of‑Thoughts**: hierarchical planning and backtracking.
- [ ] **Tool Use**: design API wrappers, chain calls, sandboxed execution.

- [ ] **Assignment:** Create an agent that solves multi-step math problems using external calculator API.

---

## Phase 11: Safety, Alignment & Evaluation

- [ ] **Bias Audits**: define demographic slices, run fairness metrics.
- [ ] **Adversarial Testing**: paraphrase and prompt‑injection attacks.
- [ ] **Explainability**: attention visualization, SHAP/LIME for token attribution.
- [ ] **Benchmarks**: MMLU, BIG-Bench, SafetyBench.

- [ ] **Deliverable:** Publish an evaluation report covering ethics, fairness, and robustness.

---

## Phase 12: Deployment & MLOps

- [ ] **Model Export**: ONNX, TorchScript, TensorRT conversion.
- [ ] **Inference Engine**: integrate FlashAttention, Triton kernels.
- [ ] **Serving**: REST/gRPC microservices, autoscaling, A/B testing.
- [ ] **Monitoring & CI/CD**: drift detection, alerting, reproducible pipelines.

- [ ] **Project:** Deploy a full LLM‑powered web service with monitoring dashboard.

---

## Phase 13: Cutting‑Edge Research & Future Directions

- [ ] **Ultra‑Long Context (1M+ tokens)**: research RoPE2, extended training regimes.
- [ ] **Dynamic Computation**: conditional computation, early exit strategies.
- [ ] **Emergent Architectures**: retrieval‑augmented expert networks, hypernetworks.
- [ ] **Synthetic Data & Continual Learning**: self‑play, model‑generated corpora.

---

## ⚙️ Usage & Contributions

1. Pick a phase from the ToC.
2. Follow `docs/phase-<n>.md` for detailed instructions.
3. Implement code in `src/phase_<n>/`.
4. Open issues & submit PRs for enhancements.

👥 **Contributions Welcome**

- Suggest new modules or improvements via issues.
- PRs for code, docs, examples, and resources.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
