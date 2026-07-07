# Phase 7 — Encoder-Decoder Architecture

Welcome to Phase 7 of your NLP learning journey! This phase is entirely dedicated to the **Encoder-Decoder (Seq2Seq)** paradigm — the critical architectural bridge between simple recurrent models and the Transformer revolution.

Understanding the encoder-decoder framework deeply — including its training techniques, decoding strategies, and fundamental limitations — provides the essential context for why the Attention Mechanism was invented.

## 📁 Folder Structure & Separate Notebooks

1.  **[01_seq2seq_deep_dive.ipynb](01_seq2seq_deep_dive.ipynb)**: A conceptual, highly visual exploration of the Encoder-Decoder Seq2Seq architecture. Details the flow from encoder hidden states through the context vector to the decoder's autoregressive generation.
2.  **[02_machine_translation.ipynb](02_machine_translation.ipynb)**: Implements a full Seq2Seq Encoder-Decoder in PyTorch from scratch. Trains a model to translate short French sentences into English, demonstrating vocabulary building, sequence padding, and teacher forcing.
3.  **[03_teacher_forcing.ipynb](03_teacher_forcing.ipynb)**: Compares training with and without teacher forcing. Demonstrates convergence speed differences, the exposure bias problem, and introduces scheduled sampling as a solution.
4.  **[04_decoding_strategies.ipynb](04_decoding_strategies.ipynb)**: Implements both Greedy Decoding and Beam Search from scratch. Compares their outputs and visualizes the quality-vs-speed trade-off across different beam widths.
5.  **[05_information_bottleneck.ipynb](05_information_bottleneck.ipynb)**: Measures how translation quality degrades as input sentence length increases, proving the fixed-size context vector is an information bottleneck. Teases the Attention Mechanism as the solution.

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
