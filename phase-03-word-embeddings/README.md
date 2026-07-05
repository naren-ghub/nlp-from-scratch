# Phase 3 — Word Embeddings: CBOW & Skip-gram

## 1. What problem does this phase solve?
Traditional vector representations like Bag of Words and TF-IDF represent words as discrete dimensions, creating a sparse coordinate system. This results in the "vocabulary mismatch" problem where synonyms (e.g. "dog" and "canine") share zero similarity because their character sequences differ. This phase introduces dense Word Embeddings, which map words to continuous low-dimensional spaces based on co-occurrence context, capturing semantic and syntactic connections geometrically.

## 2. Key Concepts
* **Vocabulary Mismatch**: The critical limitation where keyword/frequency systems fail to relate documents that use different words to convey the same meaning.
* **Sliding Context Window**: A technique that moves a window across a sequence of tokens to extract input-target context word pairs.
* **Continuous Bag of Words (CBOW)**: A Word2Vec architecture trained to predict a target center word given its surrounding context terms.
* **Skip-gram**: A Word2Vec architecture trained to predict surrounding context words given a single target input word.
* **Vector Arithmetic & Analogies**: The linear offset properties of embedding dimensions that allow models to solve analogies (like $v_{king} - v_{man} + v_{woman} \approx v_{queen}$).
* **PCA & t-SNE**: Dimensionality reduction algorithms used to project high-dimensional word vectors (e.g., 30 dimensions) onto a 2D plane for visual cluster analysis.

## 3. How to run the notebooks?
The environment uses the standard project virtual environment `.venv` with the packages `gensim`, `plotly`, and `scipy` newly installed.

To run the tutorial notebook:
1. Open a terminal in the project root.
2. Activate the virtual environment:
   ```powershell
   .venv\Scripts\activate
   ```
3. Launch Jupyter Lab:
   ```bash
   jupyter lab
   ```
4. Navigate to `phase-03-word-embeddings/01_word_embeddings.ipynb` and run all cells.

## 4. What did each task reveal?
* **Vocabulary Mismatch**: Confirmed that two documents using synonyms ("dog is quick" vs "canine is fast") yield exactly 0.0 cosine similarity under TF-IDF.
* **Sliding Context Window**: Visualized how target-context pairs shift across token sequences with margins receiving fewer context terms.
* **CBOW & Skip-gram**: Trained both architectures on a controlled toy corpus. Found that Skip-gram is more sensitive to associations on smaller datasets.
* **Vector Arithmetic**: Demonstrated that while toy models learn some semantic groupings, large corpus training (like Wikipedia) is needed to calculate perfect offsets (e.g. $king - man + woman = queen$).
* **Embedding Projections**: Plotted PCA and t-SNE coordinate maps. The scatter plots showed distinct visual clusters for animals ("dog", "canine", "cat", "feline"), royalty ("king", "queen", "prince", "princess"), and royal structures ("castle", "palace", "throne").
* **Pre-trained GloVe**: Loaded 50-dimensional Wiki-Gigaword embeddings to show immediate, highly accurate analogy outputs.

## 5. Where can I learn more?
* **Core Paper**: Mikolov et al. (2013) — *Efficient Estimation of Word Representations in Vector Space* (The landmark paper introducing Word2Vec).
* **Pre-trained Resources**: Gensim Downloader [Glove Models Reference](https://radimrehurek.com/gensim/downloader.html).
