# NLP Learning Roadmap: Foundation to Advanced

### A Hands-On, LinkedIn-Documented Journey Through Natural Language Processing

---

> This roadmap takes you from the very first principles of text processing all the way to fine-tuning large language models. Each phase is designed to be hands-on: you implement concepts from scratch before using libraries, apply them to real data, and document your learning publicly on LinkedIn. The journey spans approximately **20–22 weeks** at a sustainable pace of 8–12 hours per week.

---

## Table of Contents

1. [How to Use This Roadmap](#how-to-use-this-roadmap)
2. [Toolkit &amp; Environment Setup](#toolkit--environment-setup)
3. [Phase 1 — Tokenization &amp; Text Preprocessing](#phase-1--tokenization--text-preprocessing)
4. [Phase 2 — Bag of Words &amp; TF-IDF](#phase-2--bag-of-words--tf-idf)
5. [Phase 3 — Word Embeddings: CBOW &amp; Skip-gram](#phase-3--word-embeddings-cbow--skip-gram)
6. [Phase 4 — Classical NLP Tasks](#phase-4--classical-nlp-tasks)
7. [Phase 5 — RNNs, LSTMs &amp; Sequence Models](#phase-5--rnns-lstms--sequence-models)
8. [Phase 6 — Attention Mechanism &amp; Transformers](#phase-6--attention-mechanism--transformers)
9. [Phase 7 — BERT, GPT &amp; Pre-trained Models](#phase-7--bert-gpt--pre-trained-models)
10. [Phase 8 — RAG, Prompting &amp; Fine-tuning](#phase-8--rag-prompting--fine-tuning)
11. [LinkedIn Content Strategy](#linkedin-content-strategy)
12. [Portfolio &amp; GitHub Structure](#portfolio--github-structure)
13. [Resources &amp; References](#resources--references)

---

## How to Use This Roadmap

Follow this execution pattern for **every phase**:

1. **Implement from scratch** — write the algorithm by hand in a Jupyter notebook before touching any library. This is where real understanding is built.
2. **Reproduce with libraries** — replicate your scratch implementation using the standard ecosystem and compare outputs.
3. **Apply to real data** — pick a dataset you personally find interesting (news, movie reviews, Tamil text, tweets). Familiar data accelerates intuition.
4. **Document publicly** — write a LinkedIn post capturing what surprised you, what broke, and what you learned.
5. **Commit to GitHub** — push the notebook to your portfolio repo under the relevant phase folder before moving on.

> **Rule**: Do not skip phases to get to BERT faster. You cannot feel *why* attention matters until you have hit the limits of LSTM, and you cannot feel LSTM's value until you have lived through BoW's blindness to word order.

---

## Toolkit & Environment Setup

### Core Libraries by Phase

| Phase                        | Libraries to Install                                                       |
| ---------------------------- | -------------------------------------------------------------------------- |
| Phases 1–2 (Text basics)    | `nltk`, `spacy`, `regex`, `scikit-learn`, `pandas`, `numpy`    |
| Phase 3 (Embeddings)         | `gensim`, `fasttext`, `matplotlib`, `plotly`                       |
| Phases 4 (Classical NLP)     | `textblob`, `vaderSentiment`, `pyspellchecker`                       |
| Phases 5–6 (Deep learning)  | `torch`, `tensorflow`, `keras`                                       |
| Phase 7 (Pre-trained models) | `transformers`, `datasets`, `tokenizers`, `evaluate`               |
| Phase 8 (Advanced)           | `langchain`, `peft`, `trl`, `sentence-transformers`, `faiss-cpu` |

### What to Set Up Before Starting

Create a dedicated Python virtual environment to isolate your NLP project from your system Python. Name it clearly (e.g., `nlp-env`) and activate it every time you work on the roadmap. Install libraries phase by phase as you reach them rather than all at once — this forces you to understand what each library actually does.

Download the NLTK data bundles (tokenizers, corpora, stopwords, POS taggers) as a one-time setup step. Download the spaCy English language model separately since it is a standalone download. Keep a `requirements.txt` file updated at the end of each phase so your environment is reproducible.

### Jupyter Setup

Use JupyterLab rather than classic Jupyter for a better experience. Organize each phase into its own notebook folder. Use markdown cells generously inside notebooks — the notebook itself should read like a mini-tutorial, not just raw code.

---

## Phase 1 — Tokenization & Text Preprocessing

**Duration:** Weeks 1–2 | **LinkedIn posts:** 2–3 | **Difficulty:** ⭐☆☆☆☆

### What This Phase Is About

Tokenization is the very first step in every NLP pipeline. Before any model can process text, the raw string must be broken into discrete units called tokens. This phase also covers the cleaning operations that sit alongside tokenization in production systems. Everything you build in later phases depends on getting this right.

### Core Concepts to Master

| Concept                    | What to Understand                                                            |
| -------------------------- | ----------------------------------------------------------------------------- |
| Word tokenization          | How to split text into individual words; why whitespace alone is not enough   |
| Sentence tokenization      | How to detect sentence boundaries correctly (abbreviations, ellipsis, quotes) |
| Subword tokenization (BPE) | How rare and unknown words are handled by splitting into smaller sub-units    |
| Stopword removal           | What stopwords are, when to remove them, and when not to                      |
| Stemming                   | How suffix-chopping reduces words to approximate roots (fast but lossy)       |
| Lemmatization              | How morphological analysis reduces words to their dictionary base form        |
| Regex text cleaning        | How to use patterns to strip HTML, URLs, mentions, and special characters     |

### How to Structure the From-Scratch Implementation

Before opening any library, build a basic tokenizer yourself using nothing but Python string operations and the `re` (regex) module. The goal is not to build something production-ready — it is to understand exactly where the hard cases live.

Start with the simplest possible version: split on whitespace. Then deliberately break it with tricky examples — contractions (`don't`, `I'm`), hyphenated words (`state-of-the-art`), punctuation attached to words (`NLP.`), and sentences without spaces after periods (`Hello.World`). Each failure teaches you something the library handles silently.

Once you have identified all the edge cases, implement a more robust version using `re.findall` with a pattern that captures word characters and apostrophes separately from punctuation. Then implement a simple sentence tokenizer using punctuation-based splitting and observe where it fails (e.g., `Dr. Smith went to Washington.`).

For stemming, implement the first few rules of the Porter Stemmer by hand — strip `-ing`, `-ed`, `-ly` suffixes with basic conditions. Compare your output to the full Porter Stemmer and note where your simplification goes wrong.

### How to Structure the Library Implementation

After your scratch version, replicate everything using `nltk` and `spacy`. Work through these steps in order:

Tokenize the same test sentences with both `nltk.word_tokenize` and `spacy` and compare their outputs. Pay attention to how they handle the hard cases you found earlier. Then move to stopword removal using `nltk.corpus.stopwords` — note that the default English stopword list is opinionated and you may want to customise it for your domain.

Run both the Porter Stemmer and the Snowball Stemmer on the same set of words. Then use spaCy's lemmatizer on the same words. Build a comparison table with three columns: original word, stem, lemma. The differences will reveal each method's philosophy.

Finally, explore `tokenizers` from Hugging Face to see byte-pair encoding in action on a sample corpus. You will revisit this more deeply in Phase 7, but seeing subword tokenization at this stage creates a useful reference point.

### Hands-On Exercises

1. **Multi-language tokenization** — Tokenize text in English and one Indian language (e.g., Tamil or Hindi). Note precisely where whitespace-based splitting fails for non-Latin scripts. Explore `indicnlp` for Indian languages.
2. **Stemming vs Lemmatization comparison** — Run both on 50 diverse sentences and build a comparison table. Count how many stems produce non-dictionary words (over-stemming) vs how many lemmas miss the expected form (under-lemmatization).
3. **Noisy social media text** — Take 100 tweets from a public dataset. Write a preprocessing pipeline that removes URLs, `@mentions`, `#` symbols, and emojis while preserving the actual words. Document every decision you made.
4. **BPE from scratch** — Implement the Byte Pair Encoding merge algorithm iteratively using character-level frequency counts. Start with individual characters as your vocabulary and run 20 merge operations. Observe which character pairs get merged first.
5. **Pipeline comparison** — Run the same 20 sentences through spaCy, NLTK, and your own tokenizer. Build a table of disagreements. Investigate each disagreement and explain who is correct and why.

### LinkedIn Post Angles

- "I tokenized my first sentence today — here's why splitting text on whitespace is not as simple as it sounds." Include a side-by-side table of the same sentence tokenized three different ways.
- "Stemming chops. Lemmatization understands. I ran both on 100 words and the results surprised me." Show the most interesting disagreements between the two approaches.
- "I tried tokenizing Tamil text with an English tokenizer. Here's what broke immediately, and the lesson it taught me about language assumptions in NLP."

### Key Takeaways

- Tokenization is language-dependent and domain-dependent. There is no universal tokenizer.
- Lemmatization requires a POS tag to resolve ambiguity correctly (`lead` the metal vs `lead` the verb).
- Subword tokenization is what modern models like BERT and GPT use. Every future phase builds on this.
- Cleaning decisions made here ripple through every downstream model. Garbage in, garbage out.

---

## Phase 2 — Bag of Words & TF-IDF

**Duration:** Weeks 2–3 | **LinkedIn posts:** 2 | **Difficulty:** ⭐⭐☆☆☆

### What This Phase Is About

After tokenizing text, you need to convert it into numbers that a model can consume. Bag of Words and TF-IDF are the classical approaches. They discard word order entirely but are surprisingly effective for many classification and retrieval tasks — and understanding their limitations is what motivates everything that comes in Phase 3 and beyond.

### Core Concepts to Master

| Concept                      | What to Understand                                                                     |
| ---------------------------- | -------------------------------------------------------------------------------------- |
| One-hot encoding             | Why representing each word as a single `1` in a large vector is wasteful and limited |
| Bag of Words (BoW)           | How a document becomes a count vector over a fixed vocabulary                          |
| N-grams                      | How bigrams and trigrams partially recover local word order within BoW                 |
| TF-IDF                       | How term frequency is balanced against how common a word is across all documents       |
| Vocabulary building          | How to decide which words to include (min frequency, max features, stopwords)          |
| Sparse matrix representation | Why storing only non-zero values is essential at scale                                 |
| Cosine similarity            | How to measure similarity between two document vectors                                 |

### How to Structure the From-Scratch Implementation

Build everything from first principles using only Python dictionaries and lists. Start by building a vocabulary from a small corpus of 5–10 sentences: collect all unique words, assign each an integer index, and store the mapping in a dictionary.

Then implement a function that takes a tokenized document and your vocabulary and returns a count vector — a list of integers where each position corresponds to a word in the vocabulary and each value is how many times that word appears. Apply this to every document in your corpus and store the result as a list of lists. This is your BoW matrix.

Next implement TF-IDF from the mathematical definition. For each word in a document, compute its term frequency (count in document divided by total words in document). Then compute the inverse document frequency for each word in the vocabulary (log of total documents divided by documents containing that word). Multiply TF × IDF for each term. Compare your output to the library version and debug any discrepancies — they will teach you about edge cases like smoothing and normalization.

For n-grams, modify your vocabulary builder to generate consecutive word pairs (bigrams) in addition to single words, and observe how dramatically the vocabulary size grows.

### How to Structure the Library Implementation

Use `scikit-learn`'s `CountVectorizer` and `TfidfVectorizer`. Go through these experiments in sequence:

Fit `CountVectorizer` on a corpus of at least 20 documents and inspect the learned vocabulary. Experiment with `max_features`, `min_df`, and `max_df` parameters and observe how they prune the vocabulary. Fit `TfidfVectorizer` with `ngram_range=(1, 2)` and compare the resulting feature space to unigram-only.

The most important exercise here: print the sparse matrix as a dense array and read it visually. Understand what each row and column means. Then compute pairwise cosine similarity between all document vectors and build a similarity matrix. Visualize it as a heatmap.

### Hands-On Exercises

1. **Document similarity search** — Embed 20 news articles using TF-IDF and compute pairwise cosine similarity. Build a similarity matrix heatmap. Identify the most and least similar pairs and verify the result makes intuitive sense.
2. **BoW classification benchmark** — Train a Multinomial Naïve Bayes classifier on the 20 Newsgroups dataset using BoW vectors. Measure accuracy as you vary vocabulary size (100, 500, 1000, 5000, 10000 words). Plot the accuracy curve and find the point of diminishing returns.
3. **N-gram impact study** — Train the same classifier with unigrams only, bigrams only, and unigrams+bigrams. Compare accuracy and training time. Note when bigrams help (short texts) vs hurt (sparse data).
4. **TF-IDF top-term extraction** — For each document in a corpus, list the top 5 words by TF-IDF score. Verify that the extracted terms are genuinely descriptive of the document's content.
5. **Vocabulary analysis** — Take a domain-specific corpus (e.g., cricket match reports). Build a TF-IDF vocabulary and compare the top 50 terms against a general English corpus. Observe how domain specificity shifts the vocabulary.

### LinkedIn Post Angles

- "BoW treats every document like a bag — no order, no context. 'Dog bites man' and 'Man bites dog' look identical. Here's a visual showing when that matters and when it doesn't."
- "I built a TF-IDF similarity heatmap of 20 news articles. The visualization immediately shows which articles cover the same topic — no model, no labels, just math."

### Key Takeaways

- BoW loses all word order — this is a fundamental limitation, not a bug to fix.
- TF-IDF is still used in production search engines and outperforms raw BoW for most retrieval tasks.
- Sparse matrices are essential: a vocabulary of 50,000 words with mostly-zero document vectors would be impossibly expensive as dense arrays.
- N-grams partially recover local context but cause the feature space to explode. Use them carefully.

---

## Phase 3 — Word Embeddings: CBOW & Skip-gram

**Duration:** Weeks 3–4 | **LinkedIn posts:** 3 | **Difficulty:** ⭐⭐⭐☆☆

### What This Phase Is About

Word embeddings solve the core failure of BoW: every word is unrelated to every other word. In embedding space, `king` and `queen` are neighbours. `Paris` and `France` are related in exactly the same way `Berlin` and `Germany` are. This is where NLP starts to feel like magic — and understanding how these representations are learned is fundamental to everything that follows.

### Core Concepts to Master

| Concept                   | What to Understand                                                                                 |
| ------------------------- | -------------------------------------------------------------------------------------------------- |
| Distributional hypothesis | Words that appear in similar contexts have similar meanings — the theoretical foundation          |
| Context window            | The number of surrounding words used to define context; how window size shapes the embedding space |
| CBOW architecture         | How the model predicts a target word from the average of its context word vectors                  |
| Skip-gram architecture    | How the model predicts each context word individually from the target word                         |
| Negative sampling         | How training is made efficient by comparing real pairs against randomly sampled noise              |
| Word2Vec                  | The landmark model that made dense embeddings practical                                            |
| GloVe                     | How co-occurrence statistics across the entire corpus are used as an alternative to local context  |
| FastText                  | How character n-grams within words enable handling of rare and misspelled words                    |
| t-SNE / PCA visualization | How high-dimensional vectors are projected into 2D for visual inspection                           |

### How to Structure the From-Scratch Implementation

The goal of building Word2Vec from scratch is to understand the two weight matrices at the heart of the model — not to build something you would use in production.

Start by building a vocabulary from a small corpus and assigning each word an integer index. Then initialize two matrices randomly: an input embedding matrix (one row per word) and an output embedding matrix (one row per word). These are the parameters the model will learn.

For the Skip-gram version: for each word in the corpus, treat it as the center word and generate training pairs with each word within the context window. For each (center, context) pair, do a forward pass — look up the center word's row in the input matrix, multiply by the output matrix, apply softmax to get a probability distribution over the vocabulary, and compute the cross-entropy loss against the context word.

Walk through the backward pass manually: compute the gradient of the loss with respect to the output matrix rows and the input matrix row for the center word. Apply gradient descent to update both matrices. Do this for dozens of (center, context) pairs and then inspect the learned vectors — are similar words becoming closer in vector space?

For CBOW, the architecture is reversed: average the input embedding vectors for all context words, then predict the center word from that average. Implement this second and compare how the two architectures differ in their training dynamics.

### How to Structure the Library Implementation

Use `gensim`'s `Word2Vec` class. Train both CBOW (`sg=0`) and Skip-gram (`sg=1`) models on the same corpus, controlling for vocabulary size, embedding dimension, window size, and number of training epochs.

After training, explore the embedding space through three lenses:

First, use `most_similar` to find nearest neighbours for specific words. Test words from different semantic categories (countries, professions, sports) and verify that neighbours are semantically coherent.

Second, test vector arithmetic. The classic `king − man + woman ≈ queen` analogy is the canonical test. Design 10 of your own analogy triples based on your domain and measure how often the arithmetic produces the correct answer.

Third, visualize. Extract vectors for 50–100 words, reduce to 2D using PCA (fast, linear) and then t-SNE (slower, non-linear but reveals cluster structure). Color-code words by semantic category and observe whether the clusters align with human intuition.

Also load pre-trained GloVe vectors and compare them to your trained-from-scratch Word2Vec on the same analogy tests. The pre-trained model will almost certainly win — this is the motivation for transfer learning in Phase 7.

### Hands-On Exercises

1. **Train on a real corpus** — Download the text8 dataset or a Wikipedia dump. Train Word2Vec. Design and test at least 15 analogy triplets. Document which analogies work, which fail, and hypothesize why.
2. **CBOW vs Skip-gram comparison** — Train both on the same corpus. For 20 query words, list the top 5 most similar words from each model. Which model performs better on rare words? Which trains faster?
3. **t-SNE cluster exploration** — Project 500 word vectors using t-SNE. Color-code them by 5 semantic categories you define (e.g., animals, countries, sports, technology, emotions). Measure how cleanly the clusters separate.
4. **Out-of-vocabulary handling** — Deliberately test a list of misspelled words and invented compound words on Word2Vec (which will fail for OOV) vs FastText (which uses character n-grams). Document the difference.
5. **Domain-specific embeddings** — Train Word2Vec on a domain-specific corpus (e.g., cricket commentary, medical abstracts, Tamil news). Compare the nearest neighbours for domain terms against pre-trained general embeddings. When do domain-specific embeddings win?

### LinkedIn Post Angles

- "king − man + woman = queen. I ran this analogy on my own trained Word2Vec model. Here's what the vector arithmetic actually looks like — and the 3 cases where it failed completely."
- "CBOW vs Skip-gram: I trained both and compared them on 20 query words. Here's the visual difference in their embedding spaces and the rule for when to use each."
- "I visualized 500 word vectors with t-SNE. Countries clustered together. Animals clustered together. Verbs clustered together — and nobody told the model what a 'category' was."

### Key Takeaways

- Word2Vec is a representation, not a model for downstream tasks. You feed its outputs into classifiers, not use it to classify directly.
- Skip-gram works better on small data and rare words. CBOW is faster to train on large corpora.
- The vector arithmetic emerges from the distributional hypothesis — not from any explicit supervision.
- Pre-trained embeddings (GloVe, FastText) are almost always better than training from scratch unless your domain is very specialized.

---

## Phase 4 — Classical NLP Tasks

**Duration:** Weeks 5–7 | **LinkedIn posts:** 4–5 | **Difficulty:** ⭐⭐⭐☆☆

### What This Phase Is About

With a text representation in hand, you can now build practical systems. This phase covers the bread-and-butter NLP tasks that power real production systems today — spam filters, sentiment dashboards, search autocorrect, and named entity taggers. You will also encounter your first proper ML training loops and evaluation metrics.

### Core Concepts to Master

| Concept                  | What to Understand                                                                               |
| ------------------------ | ------------------------------------------------------------------------------------------------ |
| Text classification      | How to assign a label to a document using a pipeline of vectorization + classifier               |
| Naïve Bayes             | How Bayes' theorem and feature independence create a probabilistic text classifier               |
| SVM for text             | Why high-dimensional sparse text vectors make SVMs particularly effective                        |
| Sentiment analysis       | The difference between lexicon-based (rule-driven) and ML-based (data-driven) approaches         |
| VADER                    | How a human-curated sentiment lexicon handles social media text and negation                     |
| Auto-correction          | How edit distance quantifies how different two strings are and how it powers spelling correction |
| Named Entity Recognition | How models identify and classify names, places, and organizations in text                        |
| POS tagging              | How grammatical role labels are assigned to each token                                           |
| Evaluation metrics       | Precision, recall, F1 score, and when accuracy alone is misleading                               |

### How to Structure the From-Scratch Implementations

**Auto-correction from scratch:** Start with the concept of edit distance. Implement the dynamic programming version of Levenshtein distance — a 2D matrix where each cell represents the minimum number of insertions, deletions, or substitutions needed to transform one string into another. Walk through a worked example on paper before implementing it.

Once you have edit distance working, build a simple spell corrector. Maintain a frequency dictionary built from a large corpus. Given a misspelled word, generate all words within edit distance 1 (one character operation away) and edit distance 2. Filter these candidates to only real words using your frequency dictionary. Return the candidate with the highest corpus frequency. Test it on 20 deliberately misspelled words.

**Naïve Bayes from scratch:** Build a classifier that counts, for each class, how often each word appears. During prediction, multiply together the per-class probabilities of each word in the input document. Use log probabilities to avoid numerical underflow. Apply Laplace smoothing (add 1 to each word count) to handle words that appear in the test set but not the training set. Test it on a spam/ham classification task.

**Sentiment lexicon approach:** Build your own small sentiment lexicon by manually assigning +1 or -1 scores to 100 common opinion words. Write a scoring function that sums the scores of all matched words in a document. Handle negation by flipping the sign of any word that follows a negation word (`not`, `never`, `no`) within a 3-word window.

### How to Structure the Library Implementations

For **text classification**: build a scikit-learn Pipeline that chains a TfidfVectorizer with a classifier (try Naïve Bayes, Logistic Regression, and Linear SVM). Fit the pipeline, predict on a held-out test set, and print the full classification report. Compare all three classifiers on the same dataset.

For **sentiment analysis**: run the same set of sentences through both your hand-built lexicon and VADER. Document the disagreements — VADER's output on the same sentences will often be better because it handles exclamation marks, capitalization, and emoticons. Then build a supervised sentiment classifier using TF-IDF + Logistic Regression on the IMDB dataset and compare its accuracy against VADER.

For **NER and POS tagging**: run spaCy on a set of news articles. Print every named entity with its label and the confidence score. Then print POS tags for the same text. Investigate at least 5 entity misclassifications — understanding why a model fails is more instructive than observing when it succeeds.

### Hands-On Exercises

1. **Spam classifier comparison** — Train Naïve Bayes, Logistic Regression, and SVM on the SMS Spam Collection dataset. Compare their precision, recall, and F1 scores. Explain in plain language why precision matters more than recall in this specific task.
2. **Custom sentiment lexicon benchmark** — Build a 200-word domain-specific sentiment lexicon for a chosen domain (e.g., movie reviews, product feedback, cricket commentary). Measure its accuracy against VADER on 100 in-domain sentences.
3. **Auto-correction benchmark** — Test your Norvig-style spell corrector against `pyspellchecker` on 200 artificially corrupted words. Score by exact-match accuracy. Analyse the error types — which mistakes does your corrector handle well vs poorly?
4. **NER entity frequency analysis** — Run spaCy NER on 50 news articles. Count entity frequency by type (PERSON, ORG, GPE, DATE). Build a bar chart of the top 20 entities per type. What does the distribution reveal about what the news covers?
5. **Multi-class classification pipeline** — Use the AG News dataset (4 categories: politics, sports, business, technology). Train a TF-IDF + SVM pipeline. Visualize the confusion matrix and identify which categories are most often confused with each other.
6. **Error analysis** — Pick any classifier from the above exercises. Pull out the 20 most confidently wrong predictions. For each one, write one sentence explaining why the model got it wrong. Patterns in errors reveal model limitations.

### LinkedIn Post Angles

- "I built a spell corrector in Python — no ML, just probability theory and edit distance. Here's the intuition behind Peter Norvig's famous 21-line solution, explained simply."
- "VADER vs a fine-tuned classifier for sentiment: I tested both on 500 tweets. Here are the cases where the rule-based approach actually wins."
- "I ran NER on 50 news articles and counted every named entity. The most-mentioned PERSON surprised me. Here's the visualization."
- "Naïve Bayes classifies text by multiplying probabilities together. The 'naïve' part is assuming every word is independent. Here's when that assumption is catastrophically wrong."

### Key Takeaways

- Simple models with good features often beat complex models with poor features.
- Precision and recall tell different stories — always choose your evaluation metric based on the cost of each type of mistake.
- Lexicon-based approaches are fast, interpretable, and require no labelled data. ML approaches are more accurate but require labelled examples.
- Error analysis on wrong predictions is the fastest way to improve a model.

---

## Phase 5 — RNNs, LSTMs & Sequence Models

**Duration:** Weeks 8–10 | **LinkedIn posts:** 3–4 | **Difficulty:** ⭐⭐⭐⭐☆

### What This Phase Is About

Classical NLP models treat every word as independent. But language is deeply sequential — `not good` means the opposite of `good`, and the meaning of a word often depends on what came 10 words before it. This phase introduces recurrent architectures that process sequences step by step and maintain memory across the sequence. It also explains the training failure that motivated LSTM's invention.

### Core Concepts to Master

| Concept                        | What to Understand                                                                             |
| ------------------------------ | ---------------------------------------------------------------------------------------------- |
| Recurrent Neural Network (RNN) | How a network feeds its own hidden state back as input at each time step                       |
| Vanishing gradient problem     | Why gradients shrink exponentially over long sequences, making early steps impossible to learn |
| Exploding gradient problem     | The opposite pathology — and why gradient clipping is the standard fix                        |
| LSTM forget gate               | How the gate learns what information to erase from memory at each step                         |
| LSTM input gate                | How the gate controls what new information to write into memory                                |
| LSTM output gate               | How the gate controls what portion of memory to expose as the hidden state                     |
| GRU                            | How two gates (reset and update) replace LSTM's three gates with fewer parameters              |
| Bidirectional RNN              | How processing the sequence in both directions gives each token access to future context       |
| Sequence-to-Sequence           | How an encoder compresses the input sequence and a decoder generates the output sequence       |

### How to Structure the From-Scratch Implementation

**RNN cell from scratch:** Start with a single RNN cell. It takes two inputs — the current word embedding and the previous hidden state — and produces one output: a new hidden state. The hidden state is computed by multiplying the concatenated input through a single weight matrix and applying tanh. Implement this cell and manually unroll it across a short sequence of 5 words.

To demonstrate the vanishing gradient problem, extend your manual unrolling to sequences of length 5, 25, and 50. After each forward pass, compute the gradient of the loss with respect to the hidden state at each time step and print those gradients. Observe how the gradient magnitude shrinks (or explodes) as you go further back in time. This is the core motivation for everything LSTM introduces.

**LSTM cell from scratch:** Implement the four gate computations explicitly. At each time step, the LSTM takes the current input, the previous hidden state, and the previous cell state. It computes the forget gate (a sigmoid that decides what to erase from the cell state), the input gate (a sigmoid that decides what to update), a candidate vector (a tanh that computes the proposed new values), and the output gate (a sigmoid that decides what to expose as the hidden state). The new cell state is the element-wise sum of the gated old cell state and the gated candidate. The new hidden state is the output gate multiplied by the tanh of the new cell state.

Run the same gradient experiment on the LSTM and compare the gradient magnitude curves to the vanilla RNN. The LSTM's gradients should remain much more stable over long sequences.

### How to Structure the Library Implementation

Use PyTorch's `nn.RNN`, `nn.LSTM`, and `nn.GRU` modules. Build a sentiment classification model by stacking an embedding layer, an LSTM layer (with bidirectionality and multiple layers), a dropout layer, and a final linear classifier head.

Structure your training loop as a function that takes a dataloader, runs forward and backward passes with gradient clipping, and returns the average loss and accuracy for the epoch. Write a separate evaluation loop that runs without gradient computation. Train for 5–10 epochs and plot the train vs validation loss curves.

After training, extract the LSTM's hidden state at each time step for a specific sentence. Visualize how the hidden state representation evolves as the model reads each word. This gives intuition about what information the model is retaining.

For Seq2Seq, build a minimal encoder-decoder where the encoder is an LSTM that produces a context vector (the final hidden state), and the decoder is another LSTM that takes the context vector and generates output tokens one by one. Use this on a simple character-level toy translation task.

### Hands-On Exercises

1. **Vanishing gradient experiment** — Train a vanilla RNN on sequences of length 5, 25, and 100. After each backward pass, record the gradient norm at each time step. Plot the gradient norm decay curve for all three sequence lengths on the same graph.
2. **Gate activation visualization** — After training an LSTM sentiment classifier, extract the forget gate activation values for each token in a test sentence. Visualize them as a heatmap — which words trigger the model to reset its memory?
3. **Character-level language model** — Train a character-level LSTM on a text corpus (Shakespeare or Tamil literature). Generate text by sampling from the output distribution at each step. Compare the generated text at epoch 1, epoch 10, and epoch 50.
4. **GRU vs LSTM comparison** — Train both architectures on the IMDB dataset with identical hyperparameters. Compare training time, final accuracy, and number of trainable parameters.
5. **Sequence-to-sequence translation** — Build a minimal Seq2Seq model on the Multi30k English-to-French dataset. Implement teacher forcing during training (feed the ground-truth target token as input to the decoder at each step). Compare output quality with and without teacher forcing.

### LinkedIn Post Angles

- "I drew the LSTM cell by hand to understand every gate. Here's my annotated diagram — the forget gate is the key insight that makes RNNs actually learn long-range dependencies."
- "I measured the actual gradient norms across a 100-step sequence in a vanilla RNN. The graph makes the vanishing gradient problem impossible to ignore. This is why LSTM was invented."
- "I trained a character-level LSTM and watched it learn to write sentences. Here's the output at epoch 1, epoch 20, and epoch 50 — the progression is remarkable."

### Key Takeaways

- The vanishing gradient problem is not a hyperparameter tuning issue — it is a fundamental mathematical property of vanilla RNNs.
- LSTM gates are not switches. They are continuous values between 0 and 1, and they are learned, not programmed.
- Bidirectional RNNs are more powerful but cannot be used for generation (you cannot look at future tokens when generating left-to-right).
- Seq2Seq architecture is the conceptual predecessor of the transformer — understanding it makes attention feel like a natural evolution.

---

## Phase 6 — Attention Mechanism & Transformers

**Duration:** Weeks 11–13 | **LinkedIn posts:** 4 | **Difficulty:** ⭐⭐⭐⭐⭐

### What This Phase Is About

Attention is the single most important idea in modern NLP. Instead of compressing an entire sequence into one fixed-size vector (the Seq2Seq bottleneck), attention lets the model look directly at any part of the input when producing each output. The Transformer architecture — built entirely from attention with no recurrence — replaced RNNs in nearly every state-of-the-art system and powers BERT, GPT, and every model in Phase 7 and beyond.

### Core Concepts to Master

| Concept                           | What to Understand                                                                                                |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Query, Key, Value                 | The three projections of the input that define what you are looking for, what is available, and what you retrieve |
| Scaled dot-product attention      | How Q×Kᵀ scores are scaled by √d_k, passed through softmax, and used to weight the Values                      |
| Why the scaling factor?           | How large dot products push softmax into saturation, killing gradients — and why √d_k fixes this                |
| Multi-head attention              | How running attention in parallel with different learned projections captures different relationship types        |
| Self-attention                    | How every token in a sequence attends to every other token in the same sequence                                   |
| Cross-attention                   | How the decoder attends to encoder outputs in a full encoder-decoder transformer                                  |
| Positional encoding               | Why positional information must be injected explicitly (transformers have no built-in sequence order)             |
| Feed-forward sublayer             | The position-wise MLP that follows attention in each transformer layer                                            |
| Add & Norm (Residual connections) | How skip connections prevent gradient vanishing in deep transformers                                              |
| Causal masking                    | How the decoder prevents any position from attending to future positions during training                          |

### How to Structure the From-Scratch Implementation

**Scaled dot-product attention:** Implement the core operation step by step. Take three matrices Q, K, and V. Compute the dot product of Q and K-transposed. Divide every element by the square root of the key dimension. Apply softmax row-wise to get the attention weight matrix. Multiply by V to get the output. Print and inspect the attention weight matrix for a toy example — it should be a probability distribution summing to 1 for each row.

**Multi-head attention:** Instead of computing attention once, split the input into multiple smaller projections using learned weight matrices. Compute scaled dot-product attention independently on each projection (each "head"). Concatenate all the head outputs and project back to the original dimension with a final weight matrix. The key insight is that each head can learn to attend to a different type of relationship (syntactic, semantic, positional).

**Positional encoding:** Implement sinusoidal positional encoding. For each position index and each dimension index, compute a sine or cosine value using a specific frequency formula. Add the resulting matrix to the word embedding matrix before feeding it into the transformer. Visualize the positional encoding matrix as a heatmap to understand what patterns the transformer uses to infer position.

**Full transformer encoder layer:** Stack multi-head self-attention, a residual connection, layer normalization, a two-layer feed-forward network, another residual connection, and another layer normalization — in exactly that order. Stack multiple encoder layers. Build a toy text classifier on top of the encoder by pooling the output and passing it through a linear head.

**Causal mask:** Implement the triangular mask needed for the decoder. At each position, set the attention scores for all future positions to negative infinity before softmax. Verify that after softmax, each position can only attend to itself and past positions.

### How to Structure the Library Implementation

Train your scratch transformer encoder on a text classification task (e.g., IMDB or AG News). This is important: use exactly the same dataset you used with the LSTM in Phase 5. Then compare accuracy, training time, and convergence speed between the two architectures. This direct comparison is one of the most instructive experiments in the entire roadmap.

Use PyTorch's built-in `nn.TransformerEncoder` as a validation check — your outputs should match when given the same inputs. Then explore the trained attention weights by hooking into intermediate layers. For a given input sentence, extract the attention weight matrix from each head and each layer and visualize them.

### Hands-On Exercises

1. **Attention weight heatmap** — After training, feed a sentence through your transformer and extract the self-attention weights from each layer. Create a heatmap where rows are the attending tokens and columns are the attended-to tokens. Identify which heads have learned interpretable patterns (e.g., one head attends to adjacent words, another to coreferent pronouns).
2. **Positional encoding visualization** — Plot the sinusoidal positional encoding matrix as a heatmap with positions on one axis and dimensions on the other. Observe that nearby positions have similar encodings and that the patterns repeat at different frequencies across dimensions.
3. **Transformer vs LSTM comparison** — Train both architectures on the same dataset with the same number of epochs. Compare: final accuracy, training time per epoch, number of parameters, and convergence speed. Document which wins and why.
4. **Masking verification** — Implement the causal mask and write a test to verify it. After applying the mask and softmax, check that for every position `i`, the attention weight from token `i` to any token `j > i` is exactly zero.
5. **Attention head specialization** — Visualize all 8 heads (if using 8) from the same sentence. Do any heads appear to specialize? Design experiments to test your hypothesis — does the pattern hold across different input sentences?

### LinkedIn Post Angles

- "Attention is just asking: which other words should I look at when processing this word? I built it from scratch and here's the math, simplified to one diagram."
- "I visualized the attention weights of my transformer on the sentence 'the animal didn't cross the street because it was tired.' See where 'it' attends — that's the model resolving a pronoun."
- "Positional encoding is the transformer's way of knowing word order. Here's why sinusoidal functions were the original elegant solution."
- "I replaced my LSTM with a Transformer on the same task. Here are the two training curves side by side. The transformer converged faster — but look at what happened in the first 2 epochs."

### Key Takeaways

- Self-attention is O(n²) in sequence length — this is its fundamental scaling limitation.
- The transformer has no recurrence and no convolution. It processes all positions simultaneously, which makes it easily parallelizable on modern GPUs.
- Residual connections and layer normalization are not optional niceties — removing them makes deep transformers untrainable.
- Everything in Phase 7 (BERT, GPT) is essentially a transformer with different pre-training objectives. Understanding the architecture here makes those models transparent.

---

## Phase 7 — BERT, GPT & Pre-trained Models

**Duration:** Weeks 14–17 | **LinkedIn posts:** 4–5 | **Difficulty:** ⭐⭐⭐⭐⭐

### What This Phase Is About

BERT and GPT are the two paradigms of modern NLP: encoder-only for understanding, decoder-only for generation. Both use the transformer architecture but differ in how they are pre-trained and what tasks they excel at. Fine-tuning these models with small amounts of labelled data achieves state-of-the-art results across virtually every NLP benchmark — and understanding why requires internalizing the concept of transfer learning.

### Core Concepts to Master

| Concept                           | What to Understand                                                                                        |
| --------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Transfer learning in NLP          | How pre-training on large unlabelled corpora creates representations that transfer to downstream tasks    |
| Masked Language Modeling (MLM)    | How BERT is trained to predict randomly masked tokens using bidirectional context                         |
| Next Sentence Prediction (NSP)    | How BERT is additionally trained to classify whether two sentences follow each other                      |
| Bidirectional vs causal attention | Why BERT can see the full sequence but GPT can only see past tokens                                       |
| Causal Language Modeling (CLM)    | How GPT is trained to predict the next token given all previous tokens                                    |
| WordPiece tokenization            | The subword algorithm used by BERT; how the `[CLS]`, `[SEP]`, and `[PAD]` special tokens work       |
| Fine-tuning                       | How a pre-trained model is adapted to a specific task by adding a small task head and training end-to-end |
| Zero-shot inference               | Using a pre-trained model for a task it was never explicitly trained on, via prompt design                |
| Few-shot inference                | Providing a small number of demonstrations in the prompt as implicit task specification                   |
| DistilBERT / RoBERTa / ALBERT     | Variants that improve on the original BERT through distillation, robustness, or parameter sharing         |

### How to Structure the Conceptual Understanding

Before writing any code, spend time understanding the difference between BERT and GPT at the architectural and objective level.

**BERT's perspective:** BERT sees the entire sequence at once (bidirectional). During pre-training, 15% of tokens are masked and the model must predict them using context from both sides. This makes BERT excellent at tasks that require understanding the full sentence — classification, NER, question answering. The `[CLS]` token's final representation aggregates the entire sequence and is used as input to the classification head during fine-tuning.

**GPT's perspective:** GPT generates text left to right (causal). During pre-training, it learns to predict the next token given all previous tokens. This makes GPT excellent at generation tasks — text completion, summarization, dialogue. It cannot look at future tokens, so its representation of each token depends only on what came before.

Understanding this distinction determines which model to reach for: BERT for understanding tasks, GPT-family for generation tasks.

### How to Structure the Fine-tuning Workflow

Use the Hugging Face `transformers` ecosystem. The workflow for every fine-tuning task follows the same three steps:

**Step 1 — Data preparation:** Load your dataset using the `datasets` library. Apply the model's tokenizer to your raw text — this handles subword tokenization, adding special tokens (`[CLS]`, `[SEP]`), truncating to the model's max sequence length, and padding shorter sequences. Store the tokenized dataset in the format the trainer expects.

**Step 2 — Model setup:** Load a pre-trained model checkpoint using `AutoModel` with the appropriate head class (e.g., `AutoModelForSequenceClassification` for binary or multi-class classification, `AutoModelForTokenClassification` for NER). Set the number of output labels. The model's transformer weights are initialized from the pre-trained checkpoint; only the head is randomly initialized.

**Step 3 — Training:** Use the `Trainer` API with a `TrainingArguments` config. Set meaningful values for batch size, number of epochs, learning rate (3e-5 to 5e-5 is typical for fine-tuning), weight decay, and warmup steps. Pass in a `compute_metrics` function that evaluates accuracy (and F1 for imbalanced classes) on the validation set at the end of each epoch.

### How to Structure the Generation and Zero-Shot Experiments

For GPT-style generation, load a pre-trained model and experiment with the generation hyperparameters: temperature (controls randomness), top-k sampling (limits the pool of next tokens), top-p / nucleus sampling (limits cumulative probability), and repetition penalty. Generate the same prompt 5 times with different temperature values and observe how output diversity changes.

For zero-shot classification, use a natural language inference model (such as `facebook/bart-large-mnli`) via the `pipeline` API. Pass your input text and a list of candidate labels. The model returns a probability score for each label without ever having seen labelled training examples for your specific task.

For few-shot prompting, write a prompt that includes 2–3 demonstrating examples before your actual query. Observe how the presence and quality of examples changes the model's output compared to zero-shot.

### Hands-On Exercises

1. **Fine-tune DistilBERT on custom data** — Choose a domain-specific text classification dataset (e.g., medical abstracts, Tamil movie reviews, product feedback). Fine-tune DistilBERT. Plot the training and validation loss curves. Report precision, recall, and F1 on the test set.
2. **BERT attention visualization** — Extract attention weights from BERT's 12 heads on a pronoun resolution example (e.g., "The trophy didn't fit in the suitcase because it was too big"). Visualize which heads correctly resolve the pronoun reference.
3. **Model size vs accuracy benchmark** — Evaluate BERT-base, DistilBERT, RoBERTa-base, and ALBERT-base on the same classification task. Build a table comparing accuracy, inference latency, and parameter count. Identify the best accuracy-to-speed tradeoff.
4. **CLS embedding clustering** — Extract the `[CLS]` token embedding from BERT for 100 sentences spanning 4 topic categories. Visualize with t-SNE. Measure whether the clusters separate correctly using a simple k-means cluster purity metric.
5. **Prompt sensitivity experiment** — Feed the same 20 questions to GPT-2 using 5 differently phrased prompts for each. Document how dramatically the output changes. This is your first empirical lesson in prompt engineering.

### LinkedIn Post Angles

- "BERT reads a sentence both ways at once. GPT predicts the next word. I fine-tuned both on the same sentiment dataset — here are the numbers and what surprised me."
- "I visualized BERT's 12 attention heads on pronoun resolution. Head 8 clearly specializes in resolving what 'it' refers to. Here's the heatmap."
- "I fine-tuned DistilBERT on Tamil movie reviews. 94% accuracy, 40% smaller than full BERT, 60% faster. Here's the training curve and what the model struggles with."
- "Zero-shot classification blew my mind: I classified sentences into 5 categories I defined on the spot — no training data, no fine-tuning. Here's how it works."

### Key Takeaways

- Fine-tuning a pre-trained model almost always outperforms training from scratch, even with very small labelled datasets.
- The `[CLS]` token is a learned aggregation mechanism — it is not magic, it is just the position BERT consistently uses for classification during pre-training.
- Zero-shot and few-shot capabilities emerge from scale and pre-training diversity — they are not explicitly programmed.
- Model distillation (DistilBERT) sacrifices a small amount of accuracy for large gains in speed and memory — often the right trade-off in production.

---

## Phase 8 — RAG, Prompting & Fine-tuning

**Duration:** Weeks 18–22 | **LinkedIn posts:** 5–6 | **Difficulty:** ⭐⭐⭐⭐⭐

### What This Phase Is About

The final phase brings everything together into production-ready NLP systems. Retrieval-Augmented Generation connects LLMs to private knowledge bases, solving the hallucination and staleness problems. Prompt engineering extracts maximum performance from a model without changing its weights. LoRA fine-tuning adapts billion-parameter models on consumer hardware by training less than 1% of parameters. This phase answers the question: how do you actually ship NLP systems in the real world?

### Core Concepts to Master

| Concept                              | What to Understand                                                                                                            |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| Retrieval-Augmented Generation (RAG) | How a retrieval step fetches relevant documents at query time and includes them in the LLM prompt                             |
| Vector databases                     | How embeddings are stored and searched efficiently using approximate nearest-neighbour indices                                |
| Semantic search vs keyword search    | How embedding-based similarity differs from TF-IDF; when each approach wins                                                   |
| FAISS                                | How Facebook's Approximate Nearest Neighbour library enables fast similarity search at scale                                  |
| Sentence transformers                | How models like `all-MiniLM-L6-v2` map whole sentences to single dense vectors                                              |
| Prompt engineering                   | The systematic craft of writing input text that reliably steers model behaviour                                               |
| Chain-of-Thought (CoT)               | How including step-by-step reasoning in a prompt improves accuracy on multi-step tasks                                        |
| Role prompting                       | How framing the model as an expert persona changes output style and accuracy                                                  |
| LoRA (Low-Rank Adaptation)           | How large weight matrices are approximated by the product of two smaller matrices, dramatically reducing trainable parameters |
| PEFT                                 | The broader family of parameter-efficient fine-tuning methods                                                                 |
| BLEU, ROUGE, BERTScore               | What each metric measures, what it misses, and when to use each                                                               |

### How to Structure the RAG Pipeline

A RAG system has five distinct components. Understand each one before wiring them together.

**Component 1 — Document chunking:** Split your knowledge base into chunks of 200–500 words with 50-word overlap between consecutive chunks. Overlap ensures that important information at chunk boundaries is not lost. Experiment with different chunk sizes — smaller chunks improve retrieval precision but lose cross-sentence context.

**Component 2 — Embedding:** Use a sentence transformer model to embed each chunk into a dense vector. Every chunk becomes a single high-dimensional vector that captures its meaning. Use a model appropriate to your language — `all-MiniLM-L6-v2` is fast and English-only; use a multilingual model for Tamil or Hindi content.

**Component 3 — Index building:** Store all chunk embeddings in a FAISS index. The index enables approximate nearest-neighbour search — given a query vector, it returns the k most similar chunk vectors in milliseconds, even for millions of documents.

**Component 4 — Retrieval:** At query time, embed the user's question using the same model. Search the FAISS index for the top 3–5 most similar chunks. These are your retrieved context documents.

**Component 5 — Augmented generation:** Construct a prompt that includes the retrieved context, a clear instruction to answer based only on the provided context, and the user's question. Pass this prompt to a language model. The model's answer is grounded in your retrieved documents rather than in its (potentially outdated or hallucinated) training memory.

Build this pipeline end-to-end on a small personal knowledge base (your NLP notes from this roadmap work perfectly). Test it with 20 questions. Document the failure modes — when does retrieval fetch the wrong context? When does the model ignore the context it was given?

### How to Structure the Prompt Engineering Study

Prompt engineering is empirical. Design a structured ablation study:

Define a fixed evaluation set of 50 questions with known correct answers. For each question, test at least 4 prompt variants: zero-shot (question only), few-shot (2–3 examples before the question), chain-of-thought (add "let's think step by step"), and role-prompted (add "you are an expert in X"). Measure accuracy for each variant. Build a table showing how accuracy changes across variants and across question types.

Key principles to test empirically:

- Does adding more few-shot examples always help, or is there a point of diminishing returns?
- Does chain-of-thought help on factual recall questions, or only on reasoning questions?
- Does the order of few-shot examples matter?
- How sensitive is output to minor rephrasing of the task instruction?

Document your findings as a personal prompt engineering guide.

### How to Structure LoRA Fine-tuning

LoRA works by adding a small pair of matrices (A and B) alongside specific weight matrices in the transformer. Only A and B are trained — the original weights are frozen. The product A×B represents the weight update, and its low rank (controlled by the `r` hyperparameter) dramatically limits the number of trainable parameters.

The fine-tuning workflow:

**Step 1 — Choose your base model:** Select a model small enough to run on your hardware. A 125M–1B parameter model is trainable on a laptop CPU or a free Colab GPU.

**Step 2 — Configure LoRA:** Decide which weight matrices to apply LoRA to (query and value projection matrices are the standard choice). Set the rank `r` (start with 4 or 8) and the scaling factor `lora_alpha` (typically 2× rank). A lower rank means fewer parameters but less expressive adaptation.

**Step 3 — Prepare your dataset:** Format it as instruction-response pairs. Each training example should have a clear instruction, optional context, and the desired response.

**Step 4 — Train and compare:** Before and after fine-tuning, prompt the model with the same 10 test queries and compare the outputs qualitatively. Also measure perplexity on a held-out validation set.

**Step 5 — Merge and save:** LoRA allows you to merge the trained A×B matrices back into the original weight matrices and save a single model file, making deployment identical to a standard model.

### How to Structure the Evaluation Framework

Build a proper evaluation pipeline before presenting results. For each metric, understand what it measures and what it misses:

**BLEU:** Measures n-gram overlap between generated text and reference text. Good for translation. Penalizes output that is too short. Fails to reward paraphrases that preserve meaning but use different words.

**ROUGE:** Measures recall-oriented overlap — how much of the reference text is covered by the generated text. Variants: ROUGE-1 (unigrams), ROUGE-2 (bigrams), ROUGE-L (longest common subsequence). Primarily used for summarization.

**BERTScore:** Uses BERT embeddings to measure semantic similarity between generated and reference text. Captures meaning beyond surface n-gram overlap. More expensive to compute but more aligned with human judgement than BLEU or ROUGE.

**Human evaluation:** For generation tasks, automated metrics only partially capture quality. Design a simple human evaluation rubric with dimensions like factual accuracy, fluency, and relevance. Score 50 outputs yourself before claiming a model is "good."

### Hands-On Exercises

1. **Personal RAG chatbot** — Index your own NLP notes (from phases 1–7). Build a question-answering interface. Ask it 30 questions. Document at least 5 failure modes and explain why each failure occurs.
2. **Prompt ablation study** — Test 5 prompt strategies on a fixed evaluation set of 50 questions. Build a results table. Write a 200-word summary of what you learned about prompting from the data.
3. **LoRA fine-tuning** — Fine-tune a small instruction-tuned model on a domain-specific dataset you curate (NLP Q&A pairs, Tamil text, product reviews). Compare before/after outputs qualitatively and with BERTScore.
4. **Chunking strategy comparison** — Build the same RAG pipeline with three different chunking strategies (fixed size, sentence boundary, paragraph boundary). Test retrieval accuracy with 20 queries and compare the strategies.
5. **End-to-end NLP project** — Build a complete application combining techniques from at least 3 phases. Options: a domain-specific semantic search engine (TF-IDF + BERT embeddings + FAISS), a sentiment-aware document summarizer (LSTM classifier + T5 summarizer), or an auto-correction + classification pipeline for noisy social media text.

### LinkedIn Post Angles

- "RAG stopped my LLM from hallucinating. I connected it to my own knowledge base and the difference was immediate. Here's the full architecture — 5 components, each with a clear responsibility."
- "I ran the same 50 questions through 5 different prompt templates. Chain-of-thought beat zero-shot by 18 accuracy points. Here's the breakdown by question type."
- "LoRA fine-tuned a model by training only 0.42% of its weights. Here's the before/after output comparison and what the parameter math actually means."
- "BLEU, ROUGE, BERTScore — three ways to measure generated text quality. Here's when each metric lies to you and when to trust it."
- "I built a Q&A chatbot over my own 8-phase NLP notes. Ask it anything from Phase 1 to Phase 8. Here's the architecture, the most interesting failure I found, and what I would change next."

### Key Takeaways

- RAG is the most practical way to connect LLMs to private, up-to-date, or domain-specific knowledge. It does not require any model training.
- Prompting is engineering, not art — test variants systematically and measure results. Intuition alone will mislead you.
- LoRA makes fine-tuning accessible: a 7B parameter model can be fine-tuned in hours on a single consumer GPU.
- No automated metric fully captures generation quality. Human evaluation, even at small scale, is irreplaceable.
- The biggest gains in production NLP often come from better data, not better models.

---

## LinkedIn Content Strategy

### The Post Formula That Works

Every post should follow this five-part structure:

1. **Hook** — a surprising finding, a counterintuitive result, a relatable failure, or a specific number that creates curiosity
2. **Context** — what you were trying to do (1–2 sentences maximum)
3. **Insight** — the thing you actually learned that others would benefit from knowing
4. **Visual** — a screenshot, heatmap, chart, confusion matrix, annotated diagram, or side-by-side comparison
5. **Question** — one specific question that invites your audience to engage from their own experience

### Posting Cadence by Phase

| Phase Range                | Total Posts | Recommended Frequency |
| -------------------------- | ----------- | --------------------- |
| Phases 1–3 (Weeks 1–4)   | 7–8 posts  | Every 3–4 days       |
| Phases 4–5 (Weeks 5–10)  | 7–8 posts  | Every 3–4 days       |
| Phases 6–7 (Weeks 11–17) | 8–9 posts  | Every 3 days          |
| Phase 8 (Weeks 18–22)     | 5–6 posts  | Every 3–5 days       |

### Series Anchoring Posts

Write these specific posts to frame the narrative of your public journey:

- **Launch post (Week 1):** "I am starting a public NLP learning journey — from tokenization to fine-tuning transformers, with every step documented here. Follow along. Post 1 of ~30."
- **Milestone post (After Phase 3):** "I just trained Word2Vec from scratch and visualized the embedding space. Here is what 300 word vectors look like in 2D — and what king − man + woman actually equals on my data."
- **Midpoint post (After Phase 6):** "I just implemented the transformer attention mechanism line by line. Here are the 3 things I did not understand until I coded them myself."
- **Completion post (After Phase 8):** "22 weeks. 8 phases. 30+ posts. Here is every concept I built, every visualization I created, and the GitHub repo with all the notebooks. NLP from scratch — done."

### Hashtag Strategy

Use `#NLPFromScratch` consistently on every post to build a discoverable series. Also use `#NLP`, `#MachineLearning`, `#DeepLearning`, and `#AILearning` for discovery. For India-specific reach, add `#TechIndia` and `#AIIndia`.

---

## Portfolio & GitHub Structure

### Repository Name: `nlp-from-scratch`

Each phase gets its own folder with a consistent internal structure:

```
nlp-from-scratch/
├── README.md                          ← Top-level overview of the entire journey
├── phase-01-tokenization/
│   ├── README.md                      ← Phase-specific concept explanation
│   ├── 01_scratch_implementation.ipynb
│   ├── 02_library_implementation.ipynb
│   ├── 03_exercises.ipynb
│   ├── data/
│   └── outputs/                       ← Saved plots, figures, exported models
├── phase-02-bow-tfidf/
├── phase-03-word-embeddings/
├── phase-04-classical-nlp/
├── phase-05-rnn-lstm/
├── phase-06-attention-transformers/
├── phase-07-bert-gpt/
├── phase-08-rag-prompting-finetuning/
└── datasets/
    └── shared/                        ← Common datasets used across multiple phases
```

### Phase README Template

Every phase folder's `README.md` should answer five questions:

1. **What problem does this phase solve?** — One paragraph explaining the limitation of the previous phase that motivated this one.
2. **What are the key concepts?** — A bullet list of 5–8 concepts, each with a one-sentence explanation in your own words.
3. **How do I run the notebooks?** — Prerequisites, install commands, and the order in which to run the notebooks.
4. **What did the exercises reveal?** — A brief summary of the most interesting finding from each exercise. This is the most valuable section — it should be updated after you complete the phase.
5. **Where can I learn more?** — 2–3 links: one paper, one blog post or tutorial, and the link to your LinkedIn post(s) for this phase.

### Top-Level README Structure

The root `README.md` is your portfolio front page. It should contain:

- A one-paragraph description of the project and its purpose
- A visual timeline or progress table showing all 8 phases
- A table linking each phase to its folder, its notebook, and its LinkedIn post
- Your key results and visualizations (the best 4–6 plots from the journey)
- Your setup instructions (one block of install commands)
- A "What I learned" section written after completing the full roadmap

---

## Resources & References

### Essential Books

| Book                                         | Authors             | Best Used For                                                          |
| -------------------------------------------- | ------------------- | ---------------------------------------------------------------------- |
| Speech and Language Processing (3rd ed.)     | Jurafsky & Martin   | Comprehensive theoretical foundation — a reference, not a linear read |
| Natural Language Processing with Python      | Bird, Klein & Loper | Hands-on NLTK work in Phases 1–4                                      |
| Deep Learning for NLP                        | Palash Goyal et al. | Bridge from classical to deep methods                                  |
| Transformers for Natural Language Processing | Denis Rothman       | Practical Hugging Face implementation in Phases 6–8                   |

### Papers to Read in Order

Read these papers at the phase they correspond to — not all at once at the start.

1. **Phase 3** — Mikolov et al. (2013) — *Efficient Estimation of Word Representations in Vector Space* — the Word2Vec paper
2. **Phase 5** — Hochreiter & Schmidhuber (1997) — *Long Short-Term Memory* — the original LSTM paper
3. **Phase 6** — Vaswani et al. (2017) — *Attention Is All You Need* — the transformer paper
4. **Phase 7** — Devlin et al. (2018) — *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*
5. **Phase 7** — Radford et al. (2019) — *Language Models are Unsupervised Multitask Learners* — GPT-2
6. **Phase 8** — Lewis et al. (2020) — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
7. **Phase 8** — Hu et al. (2021) — *LoRA: Low-Rank Adaptation of Large Language Models*

### Online Courses

| Course                             | Platform              | Best For                             |
| ---------------------------------- | --------------------- | ------------------------------------ |
| Stanford CS224N                    | YouTube (free)        | Deep theoretical grounding in NLP    |
| Hugging Face NLP Course            | huggingface.co (free) | Practical Phases 6–8 implementation |
| fast.ai Practical Deep Learning    | fast.ai (free)        | Intuitive deep learning approach     |
| deeplearning.ai NLP Specialization | Coursera (paid)       | Structured coverage of all phases    |

### Datasets by Phase

| Dataset             | Task                       | How to Load                             |
| ------------------- | -------------------------- | --------------------------------------- |
| SMS Spam Collection | Spam detection             | Kaggle / UCI repository                 |
| IMDB Reviews        | Sentiment analysis         | `load_dataset("imdb")`                |
| AG News             | Topic classification       | `load_dataset("ag_news")`             |
| 20 Newsgroups       | Multi-class classification | `sklearn.datasets.fetch_20newsgroups` |
| CoNLL-2003          | Named entity recognition   | `load_dataset("conll2003")`           |
| Multi30k            | English-French translation | `load_dataset("bentrevett/multi30k")` |
| SQuAD v1.1          | Question answering         | `load_dataset("rajpurkar/squad")`     |
| Alpaca              | Instruction fine-tuning    | `load_dataset("tatsu-lab/alpaca")`    |

### Useful Blogs and Websites

- **The Illustrated Transformer** by Jay Alammar — the best visual explanation of the transformer architecture
- **The Illustrated BERT** by Jay Alammar — same treatment for BERT
- **Lilian Weng's blog** (lilianweng.github.io) — deep, rigorous write-ups on attention, RLHF, RAG, and prompting
- **Hugging Face blog** (huggingface.co/blog) — practical implementation guides for the latest models
- **Sebastian Ruder's blog** (ruder.io) — NLP research overviews and transfer learning history

---

*Last updated: June 2026 · Built as a public learning journey from tokenization to RAG*
