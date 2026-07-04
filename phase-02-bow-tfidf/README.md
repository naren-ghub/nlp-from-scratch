# Phase 2 — Bag of Words & TF-IDF

## 1. What problem does this phase solve?
Machine learning algorithms cannot directly process raw text characters or words. To feed text into a model, we must first convert it into a structured, numerical representation (vectors). This phase explores basic vectorization techniques from simple binary encoding to term importance weighting, explaining how documents can be converted into dense and sparse numerical spaces.

## 2. Key Concepts
* **One-Hot Encoding**: Represents each word in a vocabulary as a binary vector where only one element is active (1) and all others are inactive (0), establishing completely independent vector dimensions.
* **Bag of Words (BoW)**: Represents documents as occurrences of vocabulary frequencies, ignoring word sequence, grammar, and syntax structure.
* **N-grams**: Extends Bag of Words by capturing sequences of $N$ adjacent words (unigrams, bigrams, trigrams) to preserve local context and qualifiers like negations.
* **TF-IDF**: Weights terms dynamically by balancing how frequently a word appears in a specific document (Term Frequency) against how commonly it appears across the entire corpus (Inverse Document Frequency).
* **Cosine Similarity**: Measures the semantic distance between document vectors by calculating the cosine of the angle between them, focusing on orientation rather than magnitude to normalize length variations.

## 3. How to run the notebooks?
The environment uses the standard project virtual environment `.venv` with no additional package installations needed.

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
4. Navigate to `phase-02-bow-tfidf/01_bow_and_tfidf.ipynb` and run all cells.

## 4. What did each task reveal?
* **One-Hot Encoding**: Proved that all words are treated as mutually orthogonal (sharing exactly 0.0 cosine similarity), which completely blinds the representation to word meanings or synonyms.
* **Bag of Words**: Demonstrated that BoW is blind to sequence ordering; opposite statements like "dog bites man" and "man bites dog" map to identical document vectors.
* **N-grams**: Showed that adding bigrams and trigrams keeps negation modifiers linked (e.g. "not happy"), but significantly inflates vocabulary dimensionality and matrix sparsity.
* **TF-IDF**: Revealed how high-frequency words (like "nlp" in our test corpus) have their weights reduced because they appear across all documents, highlighting document-specific topic keywords.
* **Cosine Similarity**: Provided a normalized pairwise comparison matrix, visualized as a clean Seaborn heatmap showing the overlap between documents.

## 5. Where can I learn more?
* **Core Paper**: Jones, K. S. (1972) — *A Statistical Interpretation of Term Specificity and its Application in Retrieval* (The foundational paper introducing inverse document frequency).
* **Interactive Tutorial**: Scikit-Learn Guide on [Working with Text Data](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html).
