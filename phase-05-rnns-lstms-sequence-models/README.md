# Phase 5 — RNNs, LSTMs & Sequence Models

Welcome to Phase 5 of your NLP learning journey! This phase covers the foundational architectures of deep learning sequence models (RNNs, LSTMs, GRUs) and applies them to essential Sequence Modeling tasks (Next-Word Prediction, Sentiment Analysis, and Acoustic Speech Recognition).

We explore these models step-by-step to build both mathematical intuition and practical implementation experience using PyTorch.

## 📁 Folder Structure & Separate Notebooks

To make learning modular and structured, each core sequence modeling task has been separated into its own dedicated notebook:

1.  **[01_why_sequences_matter.ipynb](01_why_sequences_matter.ipynb)**: Proves why Bag-of-Words and TF-IDF models fail on sequence-dependent texts (due to their order-blind set representation mapping reversed sentences to identical vectors) and shows cosine similarity matrices.
2.  **[02_rnn_cell_from_scratch.ipynb](02_rnn_cell_from_scratch.ipynb)**: Explains the math of the RNN hidden state update, manually unrolls a cell step-by-step using NumPy, and walks through the step-by-step mechanics of Backpropagation Through Time (BPTT).
3.  **[03_vanishing_gradient.ipynb](03_vanishing_gradient.ipynb)**: Demonstrates the mathematical root cause of the Vanishing Gradient problem during temporal backpropagation in PyTorch. Captures and plots gradient norms over sequence lengths of 5, 25, and 100.
4.  **[04_lstm_gates_breakdown.ipynb](04_lstm_gates_breakdown.ipynb)**: Breaks down the cell state conveyor belt and gate equations of the LSTM cell (forget, input, candidate, output gates) using a custom numpy cell, including visual callouts for each gate component.
5.  **[05_gru_vs_lstm.ipynb](05_gru_vs_lstm.ipynb)**: Conceptually details the Update Gate and Reset Gate mechanism of Gated Recurrent Units (GRUs), implements a GRU cell step-by-step from scratch using NumPy, and analyzes when to use GRU vs LSTM with details on advantages and disadvantages (no model training/comparison in code).
6.  **[06_next_word_prediction.ipynb](deep-learning-tasks/06_next_word_prediction.ipynb)**: Explains next-token probability prediction and constructs a word-level LSTM language model trained on a creative sci-fi story about a robot explorer Sparky for sequence autocompletion. Includes an interactive `ipywidgets` auto-complete dashboard directly in the notebook allowing users to type and click suggestions to dynamically append predicted words.
7.  **[07_sentiment_analysis.ipynb](deep-learning-tasks/07_sentiment_analysis.ipynb)**: Implements deep sequence sentiment classification using PyTorch. Trains an LSTM classifier to distinguish positive and negative sentiment, using a sliced version of the NLTK movie reviews corpus for rapid training and experimentation.
8.  **[08_speech_recognition.ipynb](deep-learning-tasks/08_speech_recognition.ipynb)**: Simulates the acoustic mechanics of Speech Recognition. Generates audio waveforms and spectrograms using matplotlib, and conceptually demonstrates Connectionist Temporal Classification (CTC) decoding to translate acoustic frames into text.

*Every notebook includes a custom use-case summary and dedicated interactive evaluation interfaces so you can test your own custom strings and evaluate the output instantly.*

---

## 🛠️ Setup & Installation

Before running any notebooks, ensure you have the required packages installed in your environment:

```bash
pip install torch matplotlib seaborn pandas numpy nltk ipywidgets jupyterlab
```

---

## 💡 Key Takeaways & Findings

*   **Order-Blindness of BoW**: Classical statistical vector representations are completely blind to word sequence. Sentences with identical word vocabularies but opposite meanings (e.g. `"dog bites man"` vs. `"man bites dog"`) yield identical BoW representations (cosine similarity = 1.0) because they discard spatial structure.
*   **The Hidden State Hub**: RNNs process tokens sequentially, carrying past context forward using a recurrent hidden state equation: $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$.
*   **BPTT & Vanishing Gradients**: Because Vanilla RNNs propagate gradients back in time using repeated matrix multiplications, early sequence steps suffer from exponential decay when weight spectral radius is < 1. For a sequence length of 100, early step gradient norms shrink to $10^{-15}$, rendering early tokens unlearnable.
*   **LSTM Gating System**: LSTMs prevent vanishing gradients by introducing a cell state conveyor belt $C_t$ updated via three gates:
    *   **Forget Gate ($f_t$)**: Selectively wipes out memory parts.
    *   **Input Gate ($i_t$)**: Decides what to write.
    *   **Output Gate ($o_t$)**: Decides what to expose as the hidden state $h_t$.
*   **GRU Gating Mechanism**: GRUs simplify the gating process by merging cell and hidden states, and replacing forget/input gates with a single **Update Gate ($z_t$)** alongside a **Reset Gate ($r_t$)**. This reduces parameters by ~25% compared to LSTMs, leading to faster training times and lower computational costs.
*   **Next-Word Language Models**: Text generation is formulated as a conditional probability task. By unrolling an LSTM, the model learns the statistical flow of sequences, enabling context-aware predictive text.
*   **Deep Sentiment Analysis**: Unlike basic Bag-of-Words models, LSTM-based sentiment classifiers track long-range dependencies and word order, allowing the network to understand context shifts (e.g., negations like "not good").
*   **Acoustic Modeling**: Speech recognition relies on transforming raw audio waveforms into temporal spectrogram features, which are then passed into sequence models to map time-varying frequencies to phonetic outputs.
