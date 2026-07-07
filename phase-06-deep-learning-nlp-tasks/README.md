# Phase 6 — Deep Learning NLP Tasks

Welcome to Phase 6 of your NLP learning journey! Having explored foundational sequence architectures in the previous phase, we now apply those architectures to complex, real-world Natural Language Processing tasks.

In this phase, we dive into how LSTMs are utilized for text generation, sentiment analysis, and speech recognition.

## 📁 Folder Structure & Separate Notebooks

Each major NLP task has been separated into its own dedicated notebook to keep learning modular and focused:

1.  **[01_next_word_prediction.ipynb](01_next_word_prediction.ipynb)**: Explains next-token probability prediction and constructs a word-level LSTM language model trained on a creative sci-fi story about a robot explorer Sparky. Includes an interactive widget to test sequence autocompletion.
2.  **[02_sentiment_analysis.ipynb](02_sentiment_analysis.ipynb)**: Implements deep sequence sentiment classification using PyTorch. Trains an LSTM classifier to distinguish positive and negative sentiment, using a sliced version of the NLTK movie reviews corpus for rapid training and experimentation.
3.  **[03_speech_recognition.ipynb](03_speech_recognition.ipynb)**: Simulates the acoustic mechanics of Speech Recognition. Generates audio waveforms and spectrograms using matplotlib, and conceptually demonstrates Connectionist Temporal Classification (CTC) decoding to translate acoustic frames into text.

*Every interactive notebook includes a custom user evaluation function or interactive widget, allowing you to instantly test the trained models with your own inputs.*

---

## 🛠️ Setup & Installation

Before running any notebooks, ensure you have the required packages installed in your environment:

```bash
pip install torch matplotlib seaborn numpy nltk ipywidgets
```

---

## 💡 Key Takeaways & Findings

*   **Next-Word Language Models**: Text generation is formulated as a conditional probability task. By unrolling an LSTM, the model learns the statistical flow of sequences, enabling context-aware predictive text.
*   **Deep Sentiment Analysis**: Unlike basic Bag-of-Words models, LSTM-based sentiment classifiers track long-range dependencies and word order, allowing the network to understand context shifts (e.g., negations like "not good").
*   **Acoustic Modeling**: Speech recognition relies on transforming raw audio waveforms into temporal spectrogram features, which are then passed into sequence models to map time-varying frequencies to phonetic outputs.
