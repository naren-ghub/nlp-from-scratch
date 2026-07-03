# NLP From Scratch: Project Tracker

This file serves as the official progress log for the NLP learning journey. It tracks phase status, actual start/end dates, milestone achievements, and links to published LinkedIn content.

---

## 🚦 Phase Status Tracker

| Phase | Title | Status | Start Date | End Date | LinkedIn Posts |
|:---:|:---|:---:|:---:|:---:|:---|
| **01** | [Tokenization & Preprocessing](./phase-01-tokenization/) | 🟢 Completed | 2026-06-23 | 2026-07-03 | *None* |
| **02** | [Bag of Words & TF-IDF](./phase-02-bow-tfidf/) | 🟡 In Progress | 2026-07-03 | *TBD* | *None* |
| **03** | Word Embeddings: CBOW & Skip-gram | ⚪ Not Started | *TBD* | *TBD* | *None* |
| **04** | Classical NLP Tasks | ⚪ Not Started | *TBD* | *TBD* | *None* |
| **05** | RNNs, LSTMs & Sequence Models | ⚪ Not Started | *TBD* | *TBD* | *None* |
| **06** | Attention Mechanism & Transformers | ⚪ Not Started | *TBD* | *TBD* | *None* |
| **07** | BERT, GPT & Pre-trained Models | ⚪ Not Started | *TBD* | *TBD* | *None* |
| **08** | RAG, Prompting & Fine-tuning | ⚪ Not Started | *TBD* | *TBD* | *None* |

*Status options: ⚪ Not Started | 🟡 In Progress | 🟢 Completed*

---

## 🏆 Milestones & Achievements

- **[2026-06-23]**: Journey initiated! Local Git repository set up, `.venv` created, Phase 1 directories and tracking files initialized.
- **[2026-07-03]**: Phase 1 completed! Created comprehensive tutorial notebook covering Data Cleaning, Word Tokenization, Sentence Tokenization, Stopword Removal, Stemming, Lemmatization, and Full Pipeline Comparisons. Notebook executed successfully end-to-end.
- **[2026-07-03]**: Phase 2 completed! Implemented One-Hot Encoding, Bag of Words representation, N-grams, TF-IDF calculation from scratch and library equivalent comparison, along with document pairwise Cosine Similarity matrix and Seaborn heatmap.
- **[2026-07-03]**: Phase 3 completed! Implemented word-level synonym similarity gap demo, target-context sliding windows, trained and evaluated CBOW vs Skip-gram architectures from scratch, computed gender/semantic analogies, projected coordinates using PCA/t-SNE, and loaded pre-trained Wikipedia GloVe vectors for analogy testing.

---

## 📅 Weekly Summary Log

### Week 1 (2026-06-23 to 2026-07-03)
*Focus: Phase 1 — Setup and Tokenization basics.*
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

### Week 2 (2026-07-03)
*Focus: Phase 2 — Text Representation & Vectorization.*
- **Planned Tasks**:
  - Implement One-Hot encoding and vocabulary building.
  - Explore Bag of Words (BoW) frequency count representation and sequence blind spots.
  - Implement N-gram local context mapping to preserve negations.
  - Calculate TF-IDF weights from scratch and compare with Scikit-Learn TfidfVectorizer.
  - Calculate pairwise Cosine Similarity matrix and plot dynamic heatmap.
- **Progress Notes**:
  - Created a single structured notebook `01_bow_and_tfidf.ipynb` in the `phase-02-bow-tfidf` directory containing all Phase 2 tasks.
  - Successfully executed and saved cell outputs. Created descriptive README documentation for the phase.

### Week 3 (2026-07-03)
*Focus: Phase 3 — Word Embeddings (CBOW & Skip-gram).*
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
