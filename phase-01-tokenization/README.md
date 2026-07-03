# Phase 1: Tokenization & Text Preprocessing

This phase focuses on the first step of any Natural Language Processing pipeline: converting raw unstructured text into discrete tokens and cleaning/preprocessing them for modeling.

---

## 🎯 What Problem Does This Phase Solve?

Raw text is highly unstructured and noisy. Computers cannot process sentences as continuous strings of characters without first identifying the constituent semantic units (words, subwords, or characters). Additionally, grammatical variations (conjugations, pluralizations) and noise (formatting, punctuation, web scrapings) must be normalized. This phase implements these foundational structures from scratch and compares them to standard NLP libraries.

---

## 🔑 Key Concepts to Master

1. **Regex Text Cleaning**: Crafting regular expressions to clean URLs, HTML tags, handles, hashtags, and non-alphanumeric noise.
2. **Word Tokenization**: Splitting text into individual word tokens. We study why simple whitespace-splitting fails on contractions, hyphens, and attached punctuation.
3. **Sentence Tokenization**: Detecting sentence boundaries, handling abbreviations (e.g., `Dr.`, `e.g.`), quotes, and ellipsis.
4. **Stopword Removal**: Filtering out highly frequent but low-information words, and recognizing when they should be preserved (e.g., in sentiment analysis).
5. **Stemming**: Coarse suffix-stripping rules (like the Porter Stemmer) to reduce words to approximate roots.
6. **Lemmatization**: Morphological analysis to reduce a word to its canonical dictionary base form (lemma) using lexical databases and POS-dependent contexts.

---

## 🚀 How to Run the Notebooks

1. Activate your virtual environment:
   ```bash
   # Windows
   ..\.venv\Scripts\activate
   # macOS / Linux
   source ../.venv/bin/activate
   ```
2. Launch JupyterLab:
   ```bash
   jupyter lab
   ```
3. Run the notebook:
   - `01_tokenization_and_preprocessing.ipynb`: The comprehensive notebook containing scratch implementations, library comparisons, and the full preprocessing pipeline.

---

## 🧪 Findings and Observations

* **Cleaning tradeoff**: Normalizing lowercase and stripping punctuation is lossy. It simplifies classification vocabulary but discards sentiment emphasis (`!!!`) and entity capitalization (`US` vs `us`).
* **Negation failure**: Default stopword removal strips negation words like `not`, which flips the semantic meaning of negative sentences (e.g. "not good" -> "good"). Domain-specific lists are essential.
* **Stem vs Lemma**: Stemming (Porter, Snowball) uses deterministic suffix-stripping rules and generates non-words (e.g. "studies" -> "studi"). Lemmatization uses morphological lookup and returns valid dictionary words but is computationally slower.

---

## 📚 Further Reading & Resources

- [Speech and Language Processing (3rd ed. draft) by Dan Jurafsky and James H. Martin - Chapter 2: Regular Expressions, Text Normalization, Edit Distance](https://web.stanford.edu/~jurafsky/slp3/2.pdf)
- [Porter Stemmer official documentation & algorithm details](https://tartarus.org/martin/PorterStemmer/)
