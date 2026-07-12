# Phase 8 — BERT, GPT & Pre-trained Models

Welcome to Phase 8 of the NLP From Scratch learning journey! This phase is the bridge between the Transformer architecture (Phase 7) and the modern Large Language Model (LLM) era. We dive deep into the foundational concepts, training strategies, and architectural decisions that define BERT and GPT — the two model families that transformed NLP.

---

## 🎯 Phase Objective

By the end of this phase, you will understand:
- **Why** transfer learning works and what pre-trained models actually learn
- **How** text is tokenized using subword algorithms (BPE, WordPiece, SentencePiece)
- **What** BERT and GPT are trained to do (MLM vs. CLM) and the implications of each
- **How** BERT's bidirectional encoder architecture is structured and fine-tuned
- **How** GPT's autoregressive decoder evolved from GPT-1 through GPT-4
- **When** to use BERT vs. GPT for different NLP tasks
- **How** to use the Hugging Face ecosystem for production NLP

---

## 📁 Notebooks & Structure

| # | Notebook | Topics |
|---|---|---|
| 01 | [The Transfer Learning Revolution](01_the_transfer_learning_revolution.ipynb) | Old vs. new paradigm, pre-train→fine-tune, contextual vs. static embeddings, knowledge hierarchy |
| 02 | [Subword Tokenization](02_subword_tokenization.ipynb) | OOV problem, BPE (from scratch), WordPiece, SentencePiece, tokenizer comparison |
| 03 | [Pre-training Objectives](03_pretraining_objectives.ipynb) | Self-supervised learning, MLM + 80/10/10 masking, NSP, CLM, causal masks, scaling |
| 04 | [BERT Architecture Deep Dive](04_bert_architecture_deep_dive.ipynb) | Input embeddings, [CLS] token, attention heads, BERT variants (RoBERTa, ALBERT, DistilBERT) |
| 05 | [Fine-tuning BERT](05_finetuning_bert.ipynb) | Task heads, sentiment classification, NER, Hugging Face Trainer API |
| 06 | [GPT: The Autoregressive Revolution](06_gpt_autoregressive_revolution.ipynb) | GPT-1 & GPT-2 architecture, sampling strategies (temperature, top-k, top-p), generation |
| 07 | [GPT-3, GPT-4 & Emergent Abilities](07_gpt3_gpt4_emergence.ipynb) | Scale, in-context learning, RLHF, instruction tuning, emergent capabilities |
| 08 | [BERT vs. GPT Comparison](08_bert_vs_gpt_comparison.ipynb) | Side-by-side architecture, task suitability, embedding space analysis |
| 09 | [The Hugging Face Ecosystem](09_huggingface_ecosystem.ipynb) | Hub, Pipeline API, datasets library, Trainer, model cards |
| 10 | [Capstone Application](10_capstone_end_to_end_application.ipynb) | Multi-task pipeline: classification + NER + generation + QA |

---

## 💡 Key Takeaways

* **Transfer Learning**: Pre-training on massive unlabeled text builds rich language understanding. Fine-tuning on small labeled datasets achieves state-of-the-art results with minimal data.
* **Subword Tokenization**: BPE (GPT) and WordPiece (BERT) solve the OOV problem by splitting rare words into meaningful subword units while keeping common words whole.
* **MLM vs. CLM**: The choice of pre-training objective determines the model's strengths — bidirectional MLM enables deep understanding (BERT), while unidirectional CLM enables natural generation (GPT).
* **BERT**: An Encoder-only Transformer trained with MLM and NSP. Its `[CLS]` token output aggregates sentence-level meaning for classification. 12 attention heads specialize in different linguistic phenomena.
* **GPT**: A Decoder-only Transformer trained with CLM. Its autoregressive generation mechanism scales naturally from next-token prediction to full dialogue systems.
* **Scaling**: More data + more parameters + more compute → dramatically better models. The Chinchilla scaling laws show optimal compute-data trade-offs.

---

## 🔧 Key Libraries

```bash
pip install transformers datasets evaluate torch
```

---

## 📚 Essential Papers

| Paper | Authors | Year |
|---|---|---|
| BERT: Pre-training of Deep Bidirectional Transformers | Devlin et al. | 2018 |
| Language Models are Unsupervised Multitask Learners (GPT-2) | Radford et al. | 2019 |
| RoBERTa: A Robustly Optimized BERT Pretraining Approach | Liu et al. | 2019 |
| Language Models are Few-Shot Learners (GPT-3) | Brown et al. | 2020 |
| Training language models to follow instructions (InstructGPT) | Ouyang et al. | 2022 |

---

*Phase 8 of 9 — NLP From Scratch: Foundation to Advanced*
