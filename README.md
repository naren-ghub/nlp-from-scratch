# NLP From Scratch: Foundation to Advanced

A hands-on, portfolio-grade, publicly documented learning journey through Natural Language Processing (NLP). This repository details my progress from the core principles of text processing up to advanced architectures like Transformers, RAG pipelines, and LLM fine-tuning.

For each phase in this roadmap, I follow a strict execution pattern:
1. **Implement from scratch** — write the algorithm by hand using standard Python and minimal dependencies before using external libraries.
2. **Reproduce with libraries** — replicate the functionality using modern industrial-strength NLP libraries (e.g., `nltk`, `spaCy`, `huggingface`) and compare outputs.
3. **Apply to real data** — apply both pipelines to real-world datasets to study behaviors and edge cases.
4. **Document publicly** — write LinkedIn posts summarizing insights, design trade-offs, and lessons learned.

---

## 🗺️ Learning Journey Roadmap

| Phase | Description | Focus Areas | Status | LinkedIn Post |
| :--- | :--- | :--- | :---: | :---: |
| **01** | [Tokenization & Preprocessing](./phase-01-tokenization/) | Whitespace vs. Regex, Sentence splitting, Porter & Snowball Stemmers, spaCy Lemmatization, Tokenizer comparisons | 🟢 Completed | *TBD* |
| **02** | [Bag of Words & TF-IDF](./phase-02-bow-tfidf/) | One-Hot Encoding, Bag of Words, N-grams, TF-IDF from scratch, Cosine Similarity heatmap | 🟢 Completed | *TBD* |
| **03** | [Word Embeddings: CBOW & Skip-gram](./phase-03-word-embeddings/) | CBOW & Skip-gram from scratch, Analogy arithmetic, t-SNE & PCA visualization, Pre-trained GloVe comparison | 🟢 Completed | *TBD* |
| **04** | Classical NLP Tasks | Sentiment Analysis, Text Classification, Part-of-Speech tagging, Named Entity Recognition | ⚪ Not Started | — |
| **05** | RNNs, LSTMs & Sequence Models | Recurrent units, Backpropagation through time (BPTT), LSTM gating, POS tagging with LSTMs | ⚪ Not Started | — |
| **06** | Attention Mechanism & Transformers | Seq2Seq with attention, Self-attention, Multi-head attention, Positional encoding | ⚪ Not Started | — |
| **07** | BERT, GPT & Pre-trained Models | Encoder vs. Decoder, Masked Language Modeling (MLM), Causal LM, Hugging Face Hub | ⚪ Not Started | — |
| **08** | RAG, Prompting & Fine-tuning | Retrieval-Augmented Generation, Vector DBs (FAISS), PEFT/LoRA fine-tuning | ⚪ Not Started | — |

*Status key: ⚪ Not Started | 🟡 In Progress | 🟢 Completed*

---

## 🛠️ Tech Stack & Environment

- **Languages**: Python 3.10+
- **Core Frameworks**: PyTorch, Hugging Face Ecosystem (`transformers`, `datasets`, `tokenizers`, `peft`, `trl`)
- **Classical NLP**: spaCy, NLTK, Gensim, Scikit-learn
- **Data & Visualization**: Pandas, NumPy, Matplotlib, Plotly, JupyterLab
- **Tools & Infrastructure**: Git, GitHub, Python virtual environment (`.venv`)

---

## 🚀 Setup and Installation

### 1. Clone the repository
```bash
git clone https://github.com/naren-ghub/nlp-from-scratch.git
cd nlp-from-scratch
```

### 2. Set up the virtual environment
We use Python's built-in `venv` to isolate project dependencies.
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
Install dependencies for the active phases:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📝 What I Learned (Key Takeaways)

### Phase 01: Tokenization & Preprocessing
Successfully set up the foundation of NLP text cleaning, identifying edge cases in stemmers and observing how lemmatization relies heavily on POS tagging.

### Phase 02: Bag of Words & TF-IDF
Implemented statistical vectorization methods from scratch. Explored the limitations of word order loss in BoW and how TF-IDF penalizes common filler words.

### Phase 03: Word Embeddings (CBOW & Skip-gram)
Trained dense vector representations of words. Visualized word clusters using t-SNE/PCA and tested semantic analogies using both custom and pre-trained GloVe vectors.
