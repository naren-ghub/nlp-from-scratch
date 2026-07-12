# NLP From Scratch: Project Tracker

This file serves as the official progress log for the NLP learning journey. It tracks phase status, milestone achievements, and links to published LinkedIn content.

---

## 🚦 Phase Status Tracker

| Phase | Title | Status | LinkedIn Posts |
|:---:|:---|:---:|:---|
| **01** | [Tokenization & Preprocessing](./phase-01-tokenization/) | 🟢 Completed | [🔗 Link](https://www.linkedin.com/posts/narenkumar-n-42ba162b6_phase-1-text-preprocessing-tokenization-ugcPost-7479775514522476545-m9Ul/) |
| **02** | [Bag of Words & TF-IDF](./phase-02-bow-tfidf/) | 🟢 Completed | [🔗 Link](https://www.linkedin.com/embed/feed/update/urn:li:share:7480138268760567809) |
| **03** | [Word Embeddings: CBOW & Skip-gram](./phase-03-word-embeddings/) | 🟢 Completed | [🔗 Link](https://www.linkedin.com/posts/narenkumar-n-42ba162b6_github-naren-ghubnlp-from-scratch-a-hands-on-share-7480459316794220544-w7WM/?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEv4E-MBzduN58P3eJkO12pz6d-joiMo298) |
| **04** | [Classical NLP Tasks](./phase-04-classical-nlp/) | 🟢 Completed | [🔗 Link](https://www.linkedin.com/posts/narenkumar-n-42ba162b6_classic-nlp-tasks-ugcPost-7480278723485773825-wB3P/?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEv4E-MBzduN58P3eJkO12pz6d-joiMo298) |
| **05** | [RNNs, LSTMs & Sequence Models](./phase-05-rnns-lstms-sequence-models/) | 🟢 Completed | [🔗 Link](https://www.linkedin.com/posts/narenkumar-n-42ba162b6_deep-learning-for-nlp-ugcPost-7480912326242619396-zgr1/?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEv4E-MBzduN58P3eJkO12pz6d-joiMo298) |
| **06** | [Encoder-Decoder Architecture](./phase-06-encoder-decoder/) | 🟢 Completed | [🔗 Link](https://www.linkedin.com/posts/narenkumar-n-42ba162b6_phase-6-encoder-decoder-ugcPost-7481382175255252992-BBGD/?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEv4E-MBzduN58P3eJkO12pz6d-joiMo298) |
| **07** | [Attention Mechanism & Transformers](./phase-07-attention-transformers/) | 🟢 Completed | [🔗 Link](https://www.linkedin.com/posts/narenkumar-n-42ba162b6_attention-and-transformers-ugcPost-7481920288117403649-snlm/?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEv4E-MBzduN58P3eJkO12pz6d-joiMo298) |
| **08** | BERT, GPT & Pre-trained Models | ⚪ Not Started | *None* |
| **09** | RAG, Prompting & Fine-tuning | ⚪ Not Started | *None* |
| **10** | Application & Deployment | ⚪ Not Started | *None* |

*Status options: ⚪ Not Started | 🟡 In Progress | 🟢 Completed*

---

## 🥇 Milestones & Achievements

- **[2026-06-23]**: Journey initiated! Local Git repository set up, `.venv` created, Phase 1 directories and tracking files initialized.
- **[2026-07-03]**: Phase 1 completed! Created comprehensive tutorial notebook covering Data Cleaning, Word Tokenization, Sentence Tokenization, Stopword Removal, Stemming, Lemmatization, and Full Pipeline Comparisons. Notebook executed successfully end-to-end.
- **[2026-07-03]**: Phase 2 completed! Implemented One-Hot Encoding, Bag of Words representation, N-grams, TF-IDF calculation from scratch and library equivalent comparison, along with document pairwise Cosine Similarity matrix and Seaborn heatmap.
- **[2026-07-03]**: Phase 3 completed! Implemented word-level synonym similarity gap demo, target-context sliding windows, trained and evaluated CBOW vs Skip-gram architectures from scratch, computed gender/semantic analogies, projected coordinates using PCA/t-SNE, and loaded pre-trained Wikipedia GloVe vectors for analogy testing.
- **[2026-07-05]**: Phase 4 completed! Implemented POS Tagging & Dependency Trees, NER with displacement inline highlighting, Lexicon-based Sentiment Scoring using VADER, dynamic programming Levenshtein Auto-Correction, and SMS Spam Classifier using Naïve Bayes baseline on TF-IDF features.
- **[2026-07-06]**: Phase 5 completed! Merged core sequence models and deep learning sequence tasks. Built RNN, LSTM, and GRU gates from scratch, trained next-word prediction language model with interactive autocomplete widget, trained sentiment classifier, and generated waveform spectrograms with CTC loss.
- **[2026-07-07]**: Phase 6 completed! Mapped Seq2Seq architecture concepts, analyzed Teacher Forcing vs compounding errors, compared Greedy vs. Beam Search decoding with trace examples, analyzed the context vector information bottleneck, and implemented a PyTorch English-to-French machine translation system on the real NLTK ComTrans corpus.
- **[2026-07-09]**: Phase 7 completed! Created 6 detailed conceptual and implementation notebooks for the Attention Mechanism and Transformers. Generated 18 conceptual diagrams using matplotlib, and implemented a custom PyTorch Multi-Head Self-Attention block evaluated on sequence context mapping.

---

## 📋 Summary Log

### Phase 1 — Setup and Tokenization basics.
- **Planned Tasks**: 
  - Set up Python project template and environment.
  - Implement whitespace, regex, and sentence tokenizers from scratch.
  - Explore stemmers (from scratch rules) and BPE merging.
  - Build comparisons using NLTK and spaCy libraries.
- **Progress Notes**:
  - Successfully set up the project workspace, `.gitignore`, and tracker.
  - Built a comprehensive tutorial notebook containing all 7 core preprocessing tasks.
  - Pre-cleaned messy real-world sentences, compared whitespace vs NLTK vs spaCy tokenization, evaluated Punkt sentence segmenter, demonstrated stopword removal limitations on negation, compared Porter vs Snowball stemmers, and POS-dependent spaCy lemmatization.
  - Executed the notebook end-to-end to embed all outputs correctly.

### Phase 2 — Text Representation & Vectorization.
- **Planned Tasks**:
  - Implement One-Hot encoding and vocabulary building.
  - Explore Bag of Words (BoW) frequency count representation and sequence blind spots.
  - Implement N-gram local context mapping to preserve negations.
  - Calculate TF-IDF weights from scratch and compare with Scikit-Learn TfidfVectorizer.
  - Calculate pairwise Cosine Similarity matrix and plot dynamic heatmap.
- **Progress Notes**:
  - Created a single structured notebook `01_bow_and_tfidf.ipynb` in the `phase-02-bow-tfidf` directory containing all Phase 2 tasks.
  - Successfully executed and saved cell outputs. Created descriptive README documentation for the phase.

### Phase 3 — Word Embeddings (CBOW & Skip-gram).
- **Planned Tasks**:
  - Implement synonym semantic mismatch test in TF-IDF.
  - Generate target-context sliding window training pairs.
  - Train CBOW and Skip-gram configurations using Word2Vec and compare vectors.
  - Perform analogy vector arithmetic and print syntactic/semantic queries.
  - Reduce dimension sizes using PCA/t-SNE and plot word clustering.
  - Load pre-trained Wiki GloVe embeddings and evaluate analogy accuracy.
- **Progress Notes**:
  - Created `01_word_embeddings.ipynb` in `phase-03-word-embeddings` directory.
  - Executed the notebook end-to-end to save all plots, output matrices, and simulations. Created detailed README documentation.

### Phase 4 — Classical NLP Tasks.
- **Planned Tasks**:
  - Implement Part-of-Speech (POS) Tagging and dependency tree visuals.
  - Extract Named Entities (NER) with displaCy inline highlighting.
  - Build lexicon-based sentiment analysis with VADER (evaluating negation and boosters).
  - Implement Levenshtein edit distance DP matrix and candidates scorer from scratch.
  - Build SMS Spam text classifier using Naïve Bayes baseline on TF-IDF.
- **Progress Notes**:
  - Created and ran 5 modular notebooks in the `phase-04-classical-nlp` folder.
  - Removed extractive summarization, ML sentiment classification, and next-word prediction (moved to sequence models) from Phase 4 plan, updating tracker and README.

### Phase 5 — RNNs, LSTMs & Sequence Models.
- **Planned Tasks**:
  - Prove Bag-of-Words sequence limitations on reversed sentences.
  - Implement unrolled recurrent unit with hidden states and Backpropagation Through Time (BPTT) step-by-step gradients from scratch.
  - Detail recurrent sequence mapping configurations (One-to-One, One-to-Many, Many-to-One, Many-to-Many Synced, Many-to-Many Un-synced) conceptually.
  - Simulate vanishing/exploding gradients using recurrent weight matrix eigenvalue decay.
  - Map LSTM and GRU gating mechanisms with detailed equations and flow diagrams.
  - Implement GRU gates breakdown step-by-step from scratch and analyze selection trade-offs (use cases, advantages, and disadvantages).
  - Train an LSTM next-word predictor on Sparky story with interactive `ipywidgets` autocomplete.
  - Train PyTorch LSTM/GRU sentiment classifier on movie reviews.
  - Preprocess audio to generate spectrograms and compute CTC Loss.
- **Progress Notes**:
  - Created and executed 8 sequence notebooks inside the `phase-05-rnns-lstms-sequence-models` directory.
  - Proved BoW sequence limitations, unrolled simple RNN cell, explained hidden state, detailed RNN sequence mapping types conceptually, simulated vanishing/exploding gradients, implemented LSTM/GRU gates step-by-step from scratch with selection trade-offs.
  - Trained next-word predictor with interactive widget console, trained PyTorch LSTM sentiment classifier, and plotted spectrogram representations of waveforms with CTC Loss.

### Phase 6 — Encoder-Decoder Architecture.
- **Planned Tasks**:
  - Understand Seq2Seq mapping concepts and special tokens (<PAD>, <SOS>, <EOS>).
  - Contrast training strategies (Teacher Forcing vs. Free-Running) and explain Exposure Bias.
  - Implement Greedy Decoding vs. Beam Search and analyze performance wins.
  - Mathematically define and prove the fixed-size Context Vector Information Bottleneck.
  - Train an English-to-French PyTorch GRU Seq2Seq model on the ComTrans parallel corpus.
  - Add a live interactive translator UI widgets dashboard.
- **Progress Notes**:
  - Created 4 conceptual notebooks and 1 PyTorch implementation notebook in the `phase-06-encoder-decoder` folder.
  - Verified Beam Search victory of 7 wins out of 10 test evaluations compared to local greedy traps.
  - Plotted training loss comparisons and length accuracy bottleneck decay.


