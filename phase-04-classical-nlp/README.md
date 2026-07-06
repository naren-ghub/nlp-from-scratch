# Phase 4 — Classical NLP Tasks

Welcome to Phase 4 of your NLP learning journey! This phase covers the most common classical applications of natural language processing: classification, extractive summarization, sentiment scoring, spelling correction, next word prediction, named entity extraction, and grammatical parsing.

## 📁 Folder Structure & Separate Notebooks

To make learning modular and structured, each core classical task has been separated into its own dedicated notebook:

1.  **[01_pos_tagging.ipynb](01_pos_tagging.ipynb)**: POS tagging and Dependency Parsing with spaCy. Renders syntactic dependency trees using displaCy.
2.  **[02_named_entity_recognition.ipynb](02_named_entity_recognition.ipynb)**: Identifies proper nouns (people, locations, organizations) using spaCy. Includes visual inline entity highlighting via displaCy.
3.  **[03_extractive_summarization.ipynb](03_extractive_summarization.ipynb)**: Extractive sentence scoring summarizer built from scratch. Scores sentences using normalized term frequencies.
4.  **[04_lexicon_sentiment.ipynb](04_lexicon_sentiment.ipynb)**: Lexicon-based sentiment analysis with VADER. Evaluates how negation, capitalization, and punctuation boost scores.
5.  **[05_ml_sentiment.ipynb](05_ml_sentiment.ipynb)**: Supervised sentiment classification trained on NLTK movie reviews using Logistic Regression. Inspects positive/negative word coefficients.
6.  **[06_auto_correction.ipynb](06_auto_correction.ipynb)**: Spelling correction built from scratch. Features Levenshtein edit distance DP matrix calculations, candidate generation, and frequency scoring.
7.  **[07_next_word_prediction.ipynb](07_next_word_prediction.ipynb)**: Next word prediction using an N-gram Language Model (Trigram probability transition model built on webtext corpus).
8.  **[08_text_classification.ipynb](08_text_classification.ipynb)**: SMS Spam detection using the probabilistic Naïve Bayes baseline model on TF-IDF features.

*Every notebook includes a custom use-case summary and a dedicated interactive user evaluation function (`evaluate_...`) so you can test your own custom strings and evaluate the output instantly.*

---

## 🛠️ Setup & Installation

Before running any notebooks, ensure you have the required packages installed in your environment:

```bash
pip install vaderSentiment textblob pyspellchecker nltk spacy scikit-learn pandas numpy matplotlib seaborn
python -m spacy download en_core_web_sm
```

---

## 💡 Key Takeaways & Findings

*   **Syntax & POS Tagging**: Syntactic dependency trees let us isolate the root verb (syntactic hub) of a sentence, capturing its semantic core (e.g. `subject -> verb -> object`).
*   **NER**: Extracts structured insights from raw text by tagging persons, dates, and organizations.
*   **Extractive Summarization**: Word-frequency sentence scoring is a fast, baseline method that summarises long texts without risking hallucinations.
*   **Stopword Negation Trap**: Blind stopword removal reverses sentiment polarities (e.g. `"not good"` becomes `["good"]`). Lexicon tools like VADER or custom negation filters are needed to retain negative markers.
*   **ML Coefficients**: Logistic Regression weights represent direct corpus statistical associations, revealing specific terms driving classifications.
*   **Edit Distance & Candidate Correction**: Dynamic programming efficiently computes minimum edit steps (Levenshtein distance), and corpus frequency lists guide selections among candidates.
*   **N-gram Language Modeling**: Trigrams use historical bi-gram context to calculate statistical probabilities for predicting next words, acting as the foundation for autocomplete and text generators.
