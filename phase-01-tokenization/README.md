# Phase 1: Tokenization & Text Preprocessing

This phase focuses on the first step of any Natural Language Processing pipeline: converting raw unstructured text into discrete tokens and cleaning/preprocessing them for modeling.

---

## 🎯 What Problem Does This Phase Solve?

Raw text is highly unstructured and noisy. Computers cannot process sentences as continuous strings of characters without first identifying the constituent semantic units (words, subwords, or characters). Additionally, grammatical variations (conjugations, pluralizations) and noise (formatting, punctuation, web scrapings) must be normalized. This phase implements these foundational structures from scratch and compares them to standard NLP libraries.

---

## 🔑 Key Concepts to Master

1. **Word Tokenization**: Splitting text into individual word tokens. We study why simple whitespace-splitting fails on contractions, hyphens, and attached punctuation.
2. **Sentence Tokenization**: Detecting sentence boundaries, handling abbreviations (e.g., `Dr.`, `e.g.`), quotes, and ellipsis.
3. **Subword Tokenization (Byte Pair Encoding - BPE)**: Understanding how vocabulary is constructed by merging frequent character sequences to handle out-of-vocabulary (OOV) words.
4. **Stopword Removal**: Filtering out highly frequent but low-information words, and recognizing when they should be preserved (e.g., in sentiment analysis).
5. **Stemming**: Coarse suffix-stripping rules (like the Porter Stemmer) to reduce words to approximate roots.
6. **Lemmatization**: Morphological analysis to reduce a word to its canonical dictionary base form (lemma) using lexical databases.
7. **Regex Text Cleaning**: Crafting regular expressions to clean URLs, HTML tags, handles, hashtags, and non-alphanumeric noise.

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
   - `01_tokenization_and_preprocessing.ipynb`: The comprehensive notebook containing scratch implementations, library comparisons, and hands-on exercises.

---

## 🧪 Exercise Findings

*To be updated upon completing the exercises.*

---

## 📚 Further Reading & Resources

- [Speech and Language Processing (3rd ed. draft) by Dan Jurafsky and James H. Martin - Chapter 2: Regular Expressions, Text Normalization, Edit Distance](https://web.stanford.edu/~jurafsky/slp3/2.pdf)
- [BPE original paper: Neural Machine Translation of Rare Words with Subword Units (Sennrich et al., 2015)](https://arxiv.org/abs/1508.07909)
- [Porter Stemmer official documentation & algorithm details](https://tartarus.org/martin/PorterStemmer/)
