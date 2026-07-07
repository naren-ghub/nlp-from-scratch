# Phase 7 — Encoder-Decoder Architecture

Welcome to Phase 7 of your NLP learning journey! This phase is entirely dedicated to the **Encoder-Decoder (Seq2Seq)** paradigm — the critical architectural bridge between simple recurrent models and the Transformer revolution.

Understanding the encoder-decoder framework deeply — including its training techniques, decoding strategies, and fundamental limitations — provides the essential context for why the Attention Mechanism was invented.

## 📁 Folder Structure & Separate Notebooks

1.  **[01_seq2seq_architecture.ipynb](01_seq2seq_architecture.ipynb)**: A conceptual, highly visual explanation of the Sequence-to-Sequence Encoder-Decoder architecture.
2.  **[02_teacher_forcing.ipynb](02_teacher_forcing.ipynb)**: A conceptual exploration of training Seq2Seq models, including Teacher Forcing, Exposure Bias, and Scheduled Sampling.
3.  **[03_decoding_strategies.ipynb](03_decoding_strategies.ipynb)**: A conceptual breakdown of search strategies during inference generation, comparing Greedy decoding and Beam Search.
4.  **[04_information_bottleneck.ipynb](04_information_bottleneck.ipynb)**: A conceptual explanation of why compressing variable-length sequences creates a bottleneck, transitioning to Attention.
5.  **[05_machine_translation_implementation.ipynb](05_machine_translation_implementation.ipynb)**: Complete end-to-end PyTorch implementation of French-to-English translation from scratch. Practically demonstrates training curves, a clear Greedy vs. Beam Search case study, and length-based accuracy degradation.

---

## 🛠️ Setup & Installation

Before running any notebooks, ensure you have the required packages installed in your environment:

```bash
pip install torch matplotlib numpy
```

---

## 💡 Key Takeaways & Findings

*   **Encoder-Decoder Paradigm**: The Seq2Seq architecture uses two separate RNNs: an Encoder that compresses the input sequence into a fixed-size context vector, and a Decoder that unfolds this vector into the target sequence.
*   **Machine Translation**: Translating text involves dealing with varying sequence lengths and separate vocabularies. Special tokens (`<PAD>`, `<SOS>`, `<EOS>`) are critical for standardizing shapes and controlling generation.
*   **Teacher Forcing**: Feeding the ground-truth token as the decoder's next input during training dramatically speeds up convergence, but creates exposure bias (train ≠ inference mismatch).
*   **Decoding Strategies**: Greedy decoding is fast but locally optimal. Beam search explores multiple hypotheses in parallel and finds higher-probability sequences at the cost of computation.
*   **The Information Bottleneck**: Compressing an entire sentence into a single fixed-size vector causes information loss, especially for long inputs. This limitation directly motivated the invention of the Attention Mechanism.

## 5. Where can I learn more?
* **Core Papers**: Cho et al. (2014) — *Learning Phrase Representations using RNN Encoder-Decoder* | Sutskever et al. (2014) — *Sequence to Sequence Learning with Neural Networks*.
* **Next Phase**: The Attention Mechanism & Transformers — solving the bottleneck by letting the decoder access all encoder hidden states.
