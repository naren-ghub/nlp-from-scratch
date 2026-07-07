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
| **01** | [Tokenization & Preprocessing](./phase-01-tokenization/) | Whitespace vs. Regex, Sentence splitting, Porter & Snowball Stemmers, spaCy Lemmatization, Tokenizer comparisons | 🟢 Completed | [🔗 Link](https://www.linkedin.com/posts/narenkumar-n-42ba162b6_phase-1-text-preprocessing-tokenization-ugcPost-7479775514522476545-m9Ul/) |
| **02** | [Bag of Words & TF-IDF](./phase-02-bow-tfidf/) | One-Hot Encoding, Bag of Words, N-grams, TF-IDF from scratch, Cosine Similarity heatmap | 🟢 Completed | [🔗 Link](https://www.linkedin.com/embed/feed/update/urn:li:share:7480138268760567809) |
| **03** | [Word Embeddings: CBOW & Skip-gram](./phase-03-word-embeddings/) | CBOW & Skip-gram from scratch, Analogy arithmetic, t-SNE & PCA visualization, Pre-trained GloVe comparison | 🟢 Completed | *TBD* |
| **04** | [Classical NLP Tasks](./phase-04-classical-nlp/) | POS tagging, Dependency parsing, NER, Lexicon-based Sentiment, DP Levenshtein auto-correction, SMS spam Naïve Bayes classifier | 🟢 Completed | — |
| **05** | [RNNs, LSTMs & Sequence Models](./phase-05-rnns-lstms/) | Sequence motivation (BoW limitations), RNN cell & Hidden State from scratch, Vanishing/Exploding Gradients, LSTM cell breakdown, GRU cell breakdown | 🟢 Completed | — |
| **06** | [Deep Learning NLP Tasks](./phase-06-deep-learning-nlp-tasks/) | Next-Word Prediction, Deep Sentiment Classification, Speech Recognition acoustic features | 🟢 Completed | — |
| **07** | [Encoder-Decoder Architecture](./phase-07-encoder-decoder/) | Seq2Seq deep dive, Machine Translation, Teacher Forcing, Greedy vs Beam Search, Information Bottleneck | 🟡 In Progress | — |
| **08** | Attention Mechanism & Transformers | Seq2Seq with attention, Self-attention, Multi-head attention, Positional encoding | ⚪ Not Started | — |
| **09** | BERT, GPT & Pre-trained Models | Encoder vs. Decoder, Masked Language Modeling (MLM), Causal LM, Hugging Face Hub | ⚪ Not Started | — |
| **10** | RAG, Prompting & Fine-tuning | Retrieval-Augmented Generation, Vector DBs (FAISS), PEFT/LoRA fine-tuning | ⚪ Not Started | — |

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
# Windows
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

### Phase 04: Classical NLP Tasks
Built application pipelines including rule-based POS tagging and dependency trees, NER visualization using displaCy, lexicon-based VADER sentiment analysis, dynamic programming Levenshtein auto-correction, and Naïve Bayes SMS spam classification.

### Phase 05: RNNs, LSTMs & Sequence Models
Proved the sequence-blind spots of bag-of-words representation, implemented unrolled RNN cells from scratch (detailing mapping types conceptually), simulated vanishing/exploding gradients under recurrent weight constraints, and broke down LSTM and GRU gating mechanisms step-by-step from scratch.

### Phase 06: Deep Learning NLP Tasks
Trained an LSTM word-level language model with a dynamic in-notebook widget typing suggestions interface, built a PyTorch LSTM movie review sentiment classifier, conceptually mapped Seq2Seq Encoder-Decoder architectures, built a PyTorch French-to-English machine translation system, and extracted waveform spectrogram acoustic features alongside Connectionist Temporal Classification (CTC) sequence loss.
