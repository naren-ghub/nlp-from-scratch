# NLP Learning Roadmap: Foundation to Advanced

### A Beginner-Friendly, Step-by-Step Jupyter Notebook Journey

---

> **How to read this document**
> Every phase is broken into tasks. Every task is broken into sequential steps. Each step tells you exactly **what to do**, **what to display** in your notebook output cell, and **what to infer** from what you see. Work through each step in order — never skip a display step, because reading outputs and building intuition from them is the entire point.

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Phase 1 — Tokenization &amp; Text Preprocessing](#phase-1--tokenization--text-preprocessing)
3. [Phase 2 — Bag of Words &amp; TF-IDF](#phase-2--bag-of-words--tf-idf)
4. [Phase 3 — Word Embeddings: CBOW &amp; Skip-gram](#phase-3--word-embeddings-cbow--skip-gram)
5. [Phase 4 — Classical NLP Tasks](#phase-4--classical-nlp-tasks)
6. [Phase 5 — RNNs, LSTMs &amp; Sequence Models](#phase-5--rnns-lstms--sequence-models)
7. [Phase 6 — Attention Mechanism &amp; Transformers](#phase-6--attention-mechanism--transformers)
8. [Phase 7 — BERT, GPT &amp; Pre-trained Models](#phase-7--bert-gpt--pre-trained-models)
9. [Phase 8 — RAG, Prompting &amp; Fine-tuning](#phase-8--rag-prompting--fine-tuning)
10. [LinkedIn Content Strategy](#linkedin-content-strategy)
11. [Portfolio &amp; GitHub Structure](#portfolio--github-structure)
12. [Resources &amp; References](#resources--references)

---

## Environment Setup

Install libraries phase by phase as you reach them. Do not install everything upfront.

| Phase       | Libraries to Install                                                                      |
| ----------- | ----------------------------------------------------------------------------------------- |
| Phases 1–2 | `nltk`, `spacy`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn` |
| Phase 3     | `gensim`, `plotly`                                                                    |
| Phase 4     | `vaderSentiment`, `textblob`, `pyspellchecker`                                      |
| Phases 5–6 | `torch`                                                                                 |
| Phase 7     | `transformers`, `datasets`, `evaluate`                                              |
| Phase 8     | `sentence-transformers`, `faiss-cpu`, `langchain`, `peft`, `trl`                |

After installing `nltk`, run a one-time download of all NLTK data packages. After installing `spacy`, download the English language model separately. Confirm both are working before starting Phase 1.

---

## Phase 1 — Tokenization & Text Preprocessing

**Duration:** Weeks 1–2 | **LinkedIn posts:** 2–3 | **Difficulty:** ⭐☆☆☆☆

### Overview

This is the entry point to every NLP pipeline. Raw text is messy — it contains HTML tags, URLs, punctuation, inconsistent casing, extra spaces, and noise. Before any model can process text, you must clean it and split it into meaningful units called tokens. Every downstream phase depends on this phase being done correctly.

### Sequential Task Flow

```
Raw Text → Data Cleaning → Word Tokenization → Sentence Tokenization
        → Stopword Removal → Stemming → Lemmatization → Comparison
```

---

### Task 1 — Data Cleaning

**Objective:** Transform raw, noisy text into a clean, normalized string ready for tokenization.

---

#### Step 1 — Load raw text

- **What to do:** Collect 5–10 sentences of messy real-world text. Include at least one sentence with a URL, one with HTML tags, one with numbers, one with special characters, and one in ALL CAPS.
- **Display in notebook:** Print the raw text exactly as loaded — every piece of noise visible.
- **Infer from output:** Identify every type of noise present. Write a comment in your notebook listing each one. This list becomes your cleaning checklist.

---

#### Step 2 — Lowercase the text

- **What to do:** Convert the entire text string to lowercase.
- **Display in notebook:** Print the original text and the lowercased version side by side.
- **Infer from output:** Observe that `NLP`, `nlp`, and `Nlp` now all look the same. Note that this is not always desirable — `US` (country) and `us` (pronoun) are now identical. Record this as a known limitation.

---

#### Step 3 — Remove HTML tags

- **What to do:** Use a regex pattern to strip anything that looks like an HTML tag (text enclosed in `< >`).
- **Display in notebook:** Print a before/after pair for a sentence that contained an HTML tag.
- **Infer from output:** Confirm that the tag structure is gone but the text content between tags is preserved.

---

#### Step 4 — Remove URLs

- **What to do:** Use a regex pattern to remove `http://`, `https://`, and `www.` URLs entirely.
- **Display in notebook:** Print before/after for a URL-containing sentence.
- **Infer from output:** Observe that URLs add no semantic meaning to most NLP tasks and removing them reduces vocabulary noise.

---

#### Step 5 — Remove special characters and punctuation

- **What to do:** Strip all characters that are not letters, digits, or spaces using a regex substitution.
- **Display in notebook:** Print the text before and after this step. Also print a sorted list of every unique character that was removed.
- **Infer from output:** Notice that punctuation like `!` and `?` carried some sentiment signal (exclamation marks in positive text). Record that this step may discard useful signals — a tradeoff to revisit in Phase 4.

---

#### Step 6 — Remove extra whitespace

- **What to do:** Collapse multiple consecutive spaces, tabs, and newlines into a single space. Strip leading and trailing whitespace.
- **Display in notebook:** Print a final cleaned version of all 5–10 sentences.
- **Infer from output:** The text should now be clean, lowercase, and uniform. Compare it to your original raw text. This cleaned version is what enters the tokenizer.

---

### Task 2 — Word Tokenization

**Objective:** Split a cleaned sentence into a list of individual word tokens.

---

#### Step 1 — Tokenize by whitespace splitting (scratch approach)

- **What to do:** Split the cleaned sentence using Python's built-in `.split()` method.
- **Display in notebook:** Print the resulting list of tokens and its length.
- **Infer from output:** This works for clean text. Now test it on `"it's a state-of-the-art model"` — observe how contractions and hyphenated words are handled (or mishandled).

---

#### Step 2 — Tokenize with NLTK `word_tokenize`

- **What to do:** Apply NLTK's `word_tokenize` function to the same sentences.
- **Display in notebook:** Print the token list side by side with the whitespace-split version from Step 1.
- **Infer from output:** NLTK correctly splits `it's` into `it` and `'s`, and handles punctuation as separate tokens. Count how many tokens differ between the two approaches.

---

#### Step 3 — Tokenize with spaCy

- **What to do:** Process the same sentences using spaCy's `nlp()` pipeline and extract tokens.
- **Display in notebook:** Print the spaCy token list, and for each token also print its `.is_punct` and `.is_space` attributes.
- **Infer from output:** Compare spaCy's output to NLTK's. Note which tokenizer you prefer for which sentence types. There is no universally correct answer.

---

#### Step 4 — Test on edge cases

- **What to do:** Run all three tokenizers on these deliberate edge cases: a URL that wasn't removed, an email address, a number like `1,000.00`, and a sentence ending with `...`.
- **Display in notebook:** Print a table with the input and each tokenizer's output.
- **Infer from output:** Identify which tokenizer handles each edge case best. This builds intuition for why production pipelines often combine or customize tokenizers.

---

### Task 3 — Sentence Tokenization

**Objective:** Split a paragraph or document into individual sentences.

---

#### Step 1 — Split on period (scratch approach)

- **What to do:** Split a paragraph on the `.` character.
- **Display in notebook:** Print the resulting sentence list.
- **Infer from output:** Notice that abbreviations like `Dr.` and `U.S.A.` break the simple period split incorrectly. List every sentence where the split went wrong.

---

#### Step 2 — Use NLTK `sent_tokenize`

- **What to do:** Apply NLTK's `sent_tokenize` on the same paragraph.
- **Display in notebook:** Print the sentence list with index numbers. Print the count of sentences detected.
- **Infer from output:** NLTK's Punkt tokenizer handles abbreviations much better. Identify any remaining errors — these are the hard cases (dialogue with `"`, ellipsis, etc.).

---

#### Step 3 — Compare sentence counts on a longer document

- **What to do:** Take a 3–5 paragraph news article. Run both methods and count detected sentences.
- **Display in notebook:** Print the sentence count from each method and the sentences where they disagree.
- **Infer from output:** Quantify the error rate of the naive period-split. This motivates using a trained tokenizer for real applications.

---

### Task 4 — Stopword Removal

**Objective:** Filter out high-frequency, low-information words that add noise to representations.

---

#### Step 1 — Inspect the stopword list

- **What to do:** Load NLTK's English stopword list and print it.
- **Display in notebook:** Print all stopwords sorted alphabetically. Print the total count.
- **Infer from output:** Read through the list. Notice it includes words like `not`, `no`, and `nor`. Removing these would eliminate negation signals. Record this as an important caveat.

---

#### Step 2 — Remove stopwords from a tokenized sentence

- **What to do:** Filter out any token that appears in the stopword set from a tokenized sentence.
- **Display in notebook:** Print the original token list, the filtered token list, and a list of exactly which tokens were removed.
- **Infer from output:** Count the reduction in tokens. Observe that the remaining words are the content-bearing ones. Verify that the meaning is still roughly preserved.

---

#### Step 3 — Test on a negation example

- **What to do:** Run stopword removal on `"this movie is not good at all"` and `"this movie is good"`.
- **Display in notebook:** Print both outputs after stopword removal.
- **Infer from output:** Both sentences produce nearly identical outputs after stopword removal, yet they mean opposite things. This is a critical failure mode — note it prominently in your notebook.

---

#### Step 4 — Customize the stopword list

- **What to do:** Add domain-specific words to the stopword list (e.g., for movie reviews: `film`, `movie`, `watch`). Also remove `not` from the default list to preserve negation.
- **Display in notebook:** Show the custom stopword list. Apply it to 5 sample sentences and print the results.
- **Infer from output:** Stopword lists should always be domain-tailored. The default list is a starting point, not a final answer.

---

### Task 5 — Stemming

**Objective:** Reduce words to their approximate root form by removing suffixes.

---

#### Step 1 — Apply Porter Stemmer

- **What to do:** Run Porter Stemmer on a diverse list of 20 words (include verb forms, comparative adjectives, and gerunds).
- **Display in notebook:** Print a two-column table: original word | stem.
- **Infer from output:** Notice that some stems are not real words (e.g., `argument` → `argument`, `studies` → `studi`). This is called over-stemming. Count how many stems are not valid English words.

---

#### Step 2 — Apply Snowball Stemmer

- **What to do:** Run Snowball Stemmer on the exact same 20 words.
- **Display in notebook:** Add a third column to the table: Snowball stem.
- **Infer from output:** Compare Porter vs Snowball. Snowball is generally more conservative and produces fewer non-words. Identify the words where they disagree.

---

### Task 6 — Lemmatization

**Objective:** Reduce words to their true dictionary base form using linguistic analysis.

---

#### Step 1 — Apply spaCy lemmatizer

- **What to do:** Process the same 20 words using spaCy. Extract the `.lemma_` attribute from each token.
- **Display in notebook:** Add a fourth column to the comparison table from Task 5: lemma.
- **Infer from output:** Lemmas are always real dictionary words. Compare to the stems — notice that `studies` → `study`, `better` → `good`, `was` → `be`.

---

#### Step 2 — Observe POS dependency

- **What to do:** Run spaCy on the word `lead` in two sentences: `"He will lead the team"` and `"The pipe is made of lead"`. Extract the lemma and POS tag for the word `lead` in each.
- **Display in notebook:** Print the lemma and POS for `lead` in both sentences.
- **Infer from output:** The lemma is the same word but the POS tag differs (verb vs noun). This shows that lemmatization uses grammatical context — it cannot be applied to isolated words without context.

---

### Task 7 — Full Comparison: Raw → Clean → Tokens → No Stopwords → Stemmed → Lemmatized

**Objective:** See the complete preprocessing pipeline in one view.

---

#### Step 1 — Run the full pipeline

- **What to do:** Take 3 sentences through every step: raw text → lowercase & clean → word tokens → remove stopwords → stem → lemmatize.
- **Display in notebook:** Print the text at each stage for all 3 sentences. Format it as a pipeline table with one row per stage.
- **Infer from output:** Read each row and observe how the text transforms. Ask yourself: at which stage was information lost? At which stage was noise removed? This question has no single right answer — it depends on the task.

---


## Phase 2 — Bag of Words & TF-IDF

**Duration:** Weeks 2–3 | **LinkedIn posts:** 2 | **Difficulty:** ⭐⭐☆☆☆

### Overview

After preprocessing, text must become numbers. BoW and TF-IDF are the simplest and most widely used methods. They discard word order entirely but remain competitive on many classification and retrieval tasks. More importantly, understanding their limitations is what motivates every representation method that follows.

### Sequential Task Flow

```
Preprocessed Text → One-Hot Encoding → Bag of Words → N-grams
                 → TF Computation → IDF Computation → TF-IDF
                 → Cosine Similarity → Classification Baseline
```

---

### Task 1 — One-Hot Encoding

**Objective:** Represent each word as a binary vector before moving to richer representations.

---

#### Step 1 — Build a vocabulary

- **What to do:** Take a corpus of 5–6 short sentences. Collect all unique words and assign each an integer index in alphabetical order.
- **Display in notebook:** Print the full vocabulary dictionary (word → index). Print the vocabulary size.
- **Infer from output:** Even with 5 sentences, the vocabulary is already large relative to the number of words in any single sentence. This sparsity will become important.

---

#### Step 2 — Create one-hot vectors

- **What to do:** For each word in the vocabulary, create a vector of all zeros with a single 1 at the word's index position.
- **Display in notebook:** Print the one-hot vector for 5 different words. Also print the vector length.
- **Infer from output:** Every word vector has exactly the same length (vocabulary size) and the same number of ones (exactly 1). The cosine similarity between any two different word vectors is zero — no two words are related to each other at all. This is the fundamental limitation of one-hot encoding.

---

### Task 2 — Bag of Words

**Objective:** Represent a document as a count of how many times each vocabulary word appears.

---

#### Step 1 — Build the BoW matrix from scratch

- **What to do:** For each sentence in your corpus, create a count vector: for each vocabulary word, count how many times it appears in that sentence.
- **Display in notebook:** Print the full BoW matrix as a table where rows are documents and columns are vocabulary words. Show the actual count values inside each cell.
- **Infer from output:** Most cells are zero — the matrix is sparse. Each row is a document's "fingerprint." Words that appear in multiple documents have non-zero values in multiple rows.

---

#### Step 2 — Reproduce with scikit-learn `CountVectorizer`

- **What to do:** Fit `CountVectorizer` on the same corpus and transform all documents.
- **Display in notebook:** Print the feature names (vocabulary) and the dense matrix. Confirm it matches your scratch version exactly.
- **Infer from output:** The library handles vocabulary building, indexing, and vectorization automatically. Your scratch version proved you understand what it is doing internally.

---

#### Step 3 — Demonstrate the word order blindness

- **What to do:** Create two deliberately opposite sentences: `"the food was good not bad"` and `"the food was bad not good"`.
- **Display in notebook:** Print the BoW vector for both sentences.
- **Infer from output:** Both sentences produce identical BoW vectors despite having opposite meanings. Print this observation as a markdown cell conclusion. This is the most important limitation of BoW.

---

### Task 3 — N-grams

**Objective:** Extend BoW to capture pairs and triples of consecutive words.

---

#### Step 1 — Generate bigrams manually

- **What to do:** Take a tokenized sentence and generate all consecutive word pairs (bigrams) by sliding a window of size 2.
- **Display in notebook:** Print the original token list and then the full list of bigrams extracted from it.
- **Infer from output:** A sentence of N words produces N−1 bigrams. Notice that `"not good"` is now a single feature — this partially solves the negation problem from Phase 1.

---

#### Step 2 — Build bigram and trigram BoW

- **What to do:** Use `CountVectorizer` with `ngram_range=(2,2)` for bigrams and `(3,3)` for trigrams on your corpus.
- **Display in notebook:** Print the vocabulary size for unigrams, bigrams, and trigrams separately. Print a sample of 10 features from each.
- **Infer from output:** The vocabulary explodes with n-grams. Trigrams are so specific that most features appear only once. Print the ratio: (unique bigrams) / (unique unigrams) and (unique trigrams) / (unique unigrams).

---

#### Step 3 — Compare unigram vs bigram representation

- **What to do:** Use `ngram_range=(1,2)` to get both unigrams and bigrams combined.
- **Display in notebook:** Print the total feature count. Print the 5 most frequent and 5 least frequent features.
- **Infer from output:** Most bigrams are very sparse — they appear in very few documents. Sparse features add noise more than signal for small datasets. Record this tradeoff.

---

### Task 4 — TF-IDF

**Objective:** Weight word importance by frequency within a document balanced against frequency across all documents.

---

#### Step 1 — Compute Term Frequency (TF) from scratch

- **What to do:** For a single document, count each word and divide by the total number of words in that document.
- **Display in notebook:** Print a table: word | raw count | TF score. Sort by TF descending.
- **Infer from output:** Common words like `the` and `is` have high TF. TF alone rewards common words, which is not useful.

---

#### Step 2 — Compute Inverse Document Frequency (IDF) from scratch

- **What to do:** For each word in your vocabulary, compute IDF as the log of (total documents / number of documents containing the word). Use your full corpus.
- **Display in notebook:** Print a table: word | document frequency | IDF score. Sort by IDF descending.
- **Infer from output:** Words that appear in every document (like `the`) get a very low IDF. Words that appear in only one document get a high IDF. IDF rewards rarity.

---

#### Step 3 — Compute TF-IDF

- **What to do:** Multiply TF × IDF for every word in every document.
- **Display in notebook:** Print the full TF-IDF matrix as a table. Highlight the top 3 scoring words per document row.
- **Infer from output:** The highest TF-IDF words for each document are the ones that are both frequent in that document and rare across the corpus — these are the "characteristic" words of each document.

---

#### Step 4 — Reproduce with `TfidfVectorizer`

- **What to do:** Apply scikit-learn's `TfidfVectorizer` to the same corpus. Extract the top 5 TF-IDF words per document.
- **Display in notebook:** Print the top-5 words and their scores for each document. Compare to your scratch version.
- **Infer from output:** The library adds L2 normalization by default (row vectors sum to 1). Note this difference from your scratch version. The ranking of words should still match.

---

### Task 5 — Cosine Similarity

**Objective:** Measure how similar two documents are using their TF-IDF vectors.

---

#### Step 1 — Compute cosine similarity between two documents

- **What to do:** Implement cosine similarity from scratch: dot product of two vectors divided by the product of their magnitudes.
- **Display in notebook:** Compute and print the cosine similarity between every pair of documents in your corpus.
- **Infer from output:** Similarity ranges from 0 (nothing in common) to 1 (identical). Verify intuitively: documents on similar topics should score higher than documents on different topics.

---

#### Step 2 — Visualize the similarity matrix as a heatmap

- **What to do:** Compute a full pairwise similarity matrix and plot it as a heatmap using `seaborn`.
- **Display in notebook:** Show the heatmap with document labels on both axes and similarity values in cells.
- **Infer from output:** The diagonal is always 1.0 (a document is identical to itself). Off-diagonal cells reveal clusters of related documents. Identify the most and least similar pair.

---


## Phase 3 — Word Embeddings: CBOW & Skip-gram

**Duration:** Weeks 3–4 | **LinkedIn posts:** 3 | **Difficulty:** ⭐⭐⭐☆☆

### Overview

BoW and TF-IDF treat every word as unrelated to every other word. Word embeddings fix this by placing words in a continuous vector space where distance reflects meaning. `king` and `queen` are neighbours. `Paris` and `London` are neighbours. This phase teaches you how those embeddings are learned.

### Sequential Task Flow

```
Motivation (BoW limits) → Corpus Preparation → CBOW (concept + train)
                       → Skip-gram (concept + train) → Vector Inspection
                       → Analogy Testing → Visualization (PCA → t-SNE)
                       → Pre-trained GloVe comparison
```

---

### Task 1 — Motivation: Why BoW Is Not Enough

**Objective:** Build the intuition that motivates dense vector representations.

---

#### Step 1 — Show BoW cosine similarity for synonyms

- **What to do:** Compute the cosine similarity between the BoW vectors for `"I love NLP"` and `"I enjoy NLP"`.
- **Display in notebook:** Print the similarity score.
- **Infer from output:** The score is near zero or exactly zero because `love` and `enjoy` are different vocabulary entries with no shared tokens. Yet they mean nearly the same thing. BoW cannot capture this.

---

#### Step 2 — Show what we want embeddings to do

- **What to do:** Write a markdown cell explaining the target: after training embeddings, the cosine similarity between the vectors for `love` and `enjoy` should be high, and between `love` and `table` should be low.
- **Display in notebook:** This is a conceptual cell — no code output. Draw or sketch the 2D idea manually on paper and photograph it for your LinkedIn post.
- **Infer from output:** This motivates the next three tasks. Every embedding technique is an attempt to achieve this geometric relationship.

---

### Task 2 — Corpus Preparation

**Objective:** Prepare tokenized, cleaned text in the format that Word2Vec expects.

---

#### Step 1 — Prepare and inspect the corpus

- **What to do:** Take a text corpus of at least 50 sentences (use a downloaded dataset or a domain-specific text file). Tokenize and clean each sentence into a list of lowercase words. Store as a list of lists.
- **Display in notebook:** Print the first 5 tokenized sentences. Print the total vocabulary size and the total number of tokens.
- **Infer from output:** The vocabulary size relative to the corpus size is important. If you have 200 unique words in 50 sentences, embeddings will be low quality — you need at least tens of thousands of tokens for meaningful embeddings.

---

#### Step 2 — Inspect the context windows

- **What to do:** For a window size of 2, manually list all (center word, context word) pairs from the first 3 sentences.
- **Display in notebook:** Print each pair in the format `(center, context)`.
- **Infer from output:** Count the total pairs from just 3 sentences. This is the training data that CBOW and Skip-gram learn from. The same pair appearing many times means those words co-occur frequently — the model will learn to place them near each other in embedding space.

---

### Task 3 — CBOW (Continuous Bag of Words)

**Objective:** Understand and train the CBOW architecture, which predicts a center word from its context.

---

#### Step 1 — Understand the CBOW architecture (conceptual)

- **What to do:** Write a markdown cell describing CBOW's input and output. Input: the average of all context word vectors. Output: a probability distribution over the vocabulary. The target: the center word.
- **Display in notebook:** Draw or paste a diagram of the CBOW architecture. Label the input layer (context words), averaging operation, hidden layer, and output softmax.
- **Infer from output:** CBOW is fast because averaging context vectors is cheap. But averaging loses the order of context words — it treats `"NLP is fun"` the same as `"fun is NLP"` from the center word's perspective.

---

#### Step 2 — Train CBOW using Gensim

- **What to do:** Train a `Word2Vec` model with `sg=0` (CBOW mode) on your corpus. Set `vector_size=50`, `window=3`, `min_count=1`, `epochs=100`.
- **Display in notebook:** Print the vocabulary size learned by the model. Print the shape of the embedding matrix.
- **Infer from output:** Every word in the vocabulary now has a 50-dimensional vector. The model chose these vectors to maximize its ability to predict center words from context — that is the only training signal.

---

#### Step 3 — Inspect word vectors

- **What to do:** Print the embedding vector for 3 specific words: a common word, a rare word, and a domain-specific term.
- **Display in notebook:** Print the 50 numbers for each word. Then print the minimum, maximum, and mean value across each vector.
- **Infer from output:** The raw numbers are not directly interpretable — they only become meaningful through their relationships to other vectors. This is why the next steps (similarity and analogy) are needed.

---

#### Step 4 — Find most similar words

- **What to do:** Use `model.wv.most_similar()` to find the 5 most similar words to at least 5 different query words in your vocabulary.
- **Display in notebook:** Print a table: query word | top 5 similar words with similarity scores.
- **Infer from output:** Are the similar words semantically related? For a small corpus the results may be poor — note this and explain why (insufficient training data). For a larger corpus you should see meaningful semantic clusters.

---

### Task 4 — Skip-gram

**Objective:** Train the Skip-gram architecture and compare it directly to CBOW.

---

#### Step 1 — Understand the Skip-gram architecture (conceptual)

- **What to do:** Write a markdown cell describing Skip-gram. Input: a single center word vector. Output: a probability distribution over the vocabulary for each context position. Target: each surrounding context word individually.
- **Display in notebook:** Draw the Skip-gram diagram contrasting with the CBOW diagram from the previous task.
- **Infer from output:** Skip-gram treats each (center, context) pair as a separate training example — it generates more training signal per word, especially for rare words. This is why Skip-gram tends to produce better embeddings for rare vocabulary.

---

#### Step 2 — Train Skip-gram using Gensim

- **What to do:** Train the same `Word2Vec` with `sg=1` (Skip-gram mode) using identical hyperparameters to the CBOW model.
- **Display in notebook:** Print the vocabulary size and confirm it matches the CBOW model (same corpus).
- **Infer from output:** The model structures are identical — only the training objective differs. The learned vectors should encode different aspects of word similarity.

---

#### Step 3 — Head-to-head comparison

- **What to do:** Query `most_similar` for the same 5 words on both models. Build a side-by-side comparison.
- **Display in notebook:** Print a table with query word, CBOW top-3 similar, Skip-gram top-3 similar.
- **Infer from output:** Identify at least 2 words where the models give clearly different neighbours. For which model do the neighbours make more semantic sense to you? Note that for small corpora the difference may be subtle — the real difference shows at scale.

---

### Task 5 — Vector Arithmetic and Analogies

**Objective:** Demonstrate that semantic relationships are encoded as geometric directions in embedding space.

---

#### Step 1 — Compute the classic king analogy

- **What to do:** Using your Skip-gram model, compute `king − man + woman` using vector arithmetic. Find the word with the highest cosine similarity to the resulting vector.
- **Display in notebook:** Print the computed vector's most similar words and their scores.
- **Infer from output:** If your corpus is large enough, `queen` should appear near the top. If not (small corpus), acknowledge why: analogies require sufficient co-occurrence data to encode relational directions reliably.

---

#### Step 2 — Design and test your own analogies

- **What to do:** Design 10 analogy pairs relevant to your corpus domain. Test each one.
- **Display in notebook:** Print a results table: analogy question | expected answer | model's top-1 answer | ✓ or ✗.
- **Infer from output:** Calculate your accuracy: (correct analogies) / 10. Analyse the failures — are they due to vocabulary gaps, insufficient training data, or genuinely unexpected relationships?

---

### Task 6 — Visualization

**Objective:** Project high-dimensional word vectors into 2D to visually verify semantic clustering.

---

#### Step 1 — PCA projection

- **What to do:** Select 30–50 words across at least 4 semantic categories (e.g., countries, sports, verbs, adjectives). Extract their vectors and apply PCA to project to 2 dimensions.
- **Display in notebook:** Plot a scatter plot with word labels. Color-code points by category.
- **Infer from output:** PCA is linear and fast. Observe whether same-category words cluster together. If clusters are not clear, note that PCA may collapse important non-linear structure.

---

#### Step 2 — t-SNE projection

- **What to do:** Apply t-SNE to the same word vectors with `n_components=2`, `perplexity=10` (adjust for small vocabulary).
- **Display in notebook:** Plot the t-SNE scatter plot with the same color-coding as the PCA plot. Display both plots side by side.
- **Infer from output:** t-SNE reveals local cluster structure more clearly than PCA. Clusters that were overlapping in PCA may separate. Note that t-SNE is non-deterministic — re-running gives a different layout, but clusters should remain stable.

---

### Task 7 — Pre-trained GloVe vs Trained from Scratch

**Objective:** Understand when to use pre-trained embeddings vs training your own.

---

#### Step 1 — Load GloVe vectors and test the same analogies

- **What to do:** Download `glove.6B.50d.txt`. Load the vectors for the words in your analogy test set and run the same 10 analogies.
- **Display in notebook:** Print the analogy results for GloVe alongside your trained model results.
- **Infer from output:** GloVe (trained on 6 billion tokens) almost certainly outperforms your trained-from-scratch model. Record the accuracy gap. This is the core argument for transfer learning — which becomes the entire theme of Phase 7.

---


## Phase 4 — Classical NLP Tasks

**Duration:** Weeks 5–7 | **Difficulty:** ⭐⭐⭐☆☆

### Overview

With text representations available, you can now build real NLP applications. This phase covers the most common production NLP tasks: Part-of-Speech tagging, Named Entity Recognition, Extractive Summarization, Sentiment Analysis (Lexicon and ML-based), Spelling Auto-correction, Next Word Prediction, and Text Classification.

### Sequential Task Flow

```
POS Tagging & Dependency Parsing (spaCy)
→ Named Entity Recognition (spaCy displaCy)
→ Extractive Summarization (Word-Frequency scoring from scratch)
→ Lexicon-Based Sentiment Analysis (VADER)
→ ML-Based Sentiment Analysis (TF-IDF + Logistic Regression 3-Class)
→ Auto-Correction (DP Levenshtein edit distance & Candidate ranking)
→ Next Word Prediction (Trigram LM transition probabilities)
→ Text Classification (Naïve Bayes SMS Spam detector)
```

---

### Task 1 — POS Tagging & Dependency Parsing

**Objective:** Parse a sentence to identify grammatical roles (POS tags) and syntactic relationships between words.

---

#### Step 1 — Tag parts of speech in sentences

- **What to do:** Parse 5 diverse sentences containing words with ambiguous parts of speech (e.g. "book" as verb and noun) using the `en_core_web_sm` model in spaCy.
- **Display in notebook:** Print a table showing: token | POS tag | dependency tag | head word.
- **Infer from output:** Observe how POS tagging changes depending on context. Count nouns, verbs, and adjectives.

---

#### Step 2 — Visualize the dependency tree

- **What to do:** Use `spacy.displacy.render(style="dep")` to show the dependency parse tree of a sentence.
- **Display in notebook:** The rendered tree inline.
- **Infer from output:** Trace subject -> verb -> object paths to isolate the syntactic hub of a sentence.

---

### Task 2 — Named Entity Recognition (NER)

**Objective:** Identify and classify proper nouns representing people, places, dates, and organizations in text.

---

#### Step 1 — Run spaCy NER on a text snippet

- **What to do:** Parse a news article containing organizations, dates, locations, and names using spaCy.
- **Display in notebook:** Print a table showing entity text, label, and character offsets.
- **Infer from output:** Evaluate the accuracy of the entity extraction. Spot any incorrect classifications.

---

#### Step 2 — Visualize entities with displaCy

- **What to do:** Use `spacy.displacy.render(style="ent")` to highlight entities inline.
- **Display in notebook:** Color-coded HTML markup in context.
- **Infer from output:** Assess which labels (e.g., ORG, GPE, PERSON) are most reliable and which are error-prone.

---

### Task 3 — Extractive Summarization

**Objective:** Build an extractive summarizer from scratch by scoring sentences using normalized word frequencies.

---

#### Step 1 — Implement word frequency scoring

- **What to do:** Tokenize text, calculate word frequency dict (excluding stopwords), and normalize weights by dividing by maximum frequency.
- **Display in notebook:** Print normalized frequency of top 10 content words.
- **Infer from output:** Confirm that filler words are ignored and top words capture the main theme.

---

#### Step 2 — Score and rank sentences

- **What to do:** Score sentences by summing normalized word frequencies and dividing by sentence length (optional). Extract the top N sentences as a summary.
- **Display in notebook:** Print the original text, the generated summary, and the compression ratio.
- **Infer from output:** Observe how frequency-based scoring selects representative sentences without risk of hallucination.

---

### Task 4 — Lexicon-Based Sentiment Analysis

**Objective:** Use the VADER dictionary to score sentiment intensity, accounting for punctuation, capitalization, and negation.

---

#### Step 1 — Load VADER & test words/phrases

- **What to do:** Score individual sentiment words (`good`, `bad`) and observe the compound score.
- **Display in notebook:** Print compound, positive, negative, and neutral scores.
- **Infer from output:** Observe how negation (`not good`) and intensifiers (`extremely good!`) affect VADER's compound scores.

---

#### Step 2 — Evaluate diverse inputs

- **What to do:** Score 10 custom sentences including sarcasm, negations, and double negatives.
- **Display in notebook:** Print sentence text, compound score, and mapped label (positive/negative/neutral).
- **Infer from output:** Document limitations of dictionary approaches, especially when processing sarcasm.

---

### Task 5 — ML-Based Sentiment Analysis

**Objective:** Train a 3-class Logistic Regression classifier on labeled sentences to classify sentiment.

---

#### Step 1 — Build and train the classifier

- **What to do:** Extract sentences from `movie_reviews`, label them (positive/negative/neutral) using VADER, split 80/20, vectorize using `TfidfVectorizer`, and train `LogisticRegression`.
- **Display in notebook:** Print classification report (Precision, Recall, F1) for all 3 classes.
- **Infer from output:** Note performance across positive, negative, and neutral categories.

---

#### Step 2 — Extract coefficients & test custom inputs

- **What to do:** Print the top 10 positive, negative, and neutral words according to the classifier's coefficients. Predict on custom test strings.
- **Display in notebook:** Print top coefficients and test predictions with confidence probabilities.
- **Infer from output:** Check if learned associations correspond to human intuition.

---

### Task 6 — Auto-Correction

**Objective:** Build a spelling corrector from scratch using Levenshtein distance and word frequency lists.

---

#### Step 1 — Implement Levenshtein Distance from scratch

- **What to do:** Write the 2D DP Levenshtein algorithm and calculate the distance matrix for `("speling", "spelling")`.
- **Display in notebook:** Print the full annotated 2D DP matrix.
- **Infer from output:** Trace the traceback path to find edit operations (insertions/deletions).

---

#### Step 2 — Generate candidates & score by frequency

- **What to do:** Generate edit-distance 1 and 2 edits. Filter using a vocabulary frequency dict (using `nltk.corpus.words`).
- **Display in notebook:** Print candidates and correction with the highest corpus frequency.
- **Infer from output:** Evaluate spelling corrector on misspelled words.

---

### Task 7 — Next Word Prediction

**Objective:** Train a Trigram language model to calculate transitional probabilities and predict the next word.

---

#### Step 1 — Build N-gram counts

- **What to do:** Extract trigrams from a text corpus (e.g., `webtext`). Compute frequency counts of bigram contexts and transition words.
- **Display in notebook:** Print top transition candidates and probabilities for a few bigram contexts (e.g., "of the").
- **Infer from output:** Understand how statistical word transitions model language flow.

---

#### Step 2 — Interactive transition prediction

- **What to do:** Write a function to predict the top 3 most likely next words for a user input.
- **Display in notebook:** Print user input and transition probabilities.
- **Infer from output:** Test multi-word generation by greedily appending predicted words.

---

### Task 8 — Text Classification

**Objective:** Build an SMS Spam detector using Naïve Bayes and evaluate it using confusion heatmaps.

---

#### Step 1 — Build Naïve Bayes pipeline

- **What to do:** Load a spam dataset, vectorize text with `TfidfVectorizer`, split 80/20, and train `MultinomialNB`.
- **Display in notebook:** Print classification report (F1-score for spam/ham).
- **Infer from output:** Evaluate the performance. Discuss why F1-score is more reliable than accuracy here.

---

#### Step 2 — Evaluate errors & confusion matrix

- **What to do:** Compute the confusion matrix and plot as a Seaborn heatmap. Find examples of classification disagreements.
- **Display in notebook:** Print Seaborn heatmap and sample error classifications.
- **Infer from output:** Analyze False Positives vs False Negatives.

---

## Phase 5 — RNNs, LSTMs & Sequence Models

**Duration:** Weeks 8–10 | **LinkedIn posts:** 3–4 | **Difficulty:** ⭐⭐⭐⭐☆

### Overview

Classical NLP treats every word as independent. RNNs process text as a sequence, maintaining a memory state as they read each word. LSTMs extend RNNs with gated memory that selectively remembers and forgets — solving the training failure that made vanilla RNNs impractical for long sequences.

### Sequential Task Flow

```
Motivation (BoW vs Sequence) → RNN Cell (manual unroll)
→ Vanishing Gradient Demo → LSTM Architecture (gate by gate)
→ LSTM Training (sentiment) → Gate Visualization
→ GRU Comparison → Bidirectional LSTM
```

---

### Task 1 — Motivation: Why Sequences Matter

**Objective:** Prove empirically that BoW cannot handle sequence-dependent meaning.

---

#### Step 1 — BoW failure on sequence-dependent examples

- **What to do:** Create 5 pairs of sentences with identical word sets but different word order and different meanings (e.g., "dog bites man" / "man bites dog", "not good" / "good not").
- **Display in notebook:** Print both BoW vectors for each pair side by side. Compute cosine similarity within each pair.
- **Infer from output:** The similarity for opposite-meaning sentence pairs is near 1.0. Print this as a markdown conclusion: BoW is blind to word order, and word order can completely reverse meaning.

---

### Task 2 — The Recurrent Neural Network Cell

**Objective:** Understand the RNN's hidden state mechanism by unrolling it manually.

---

#### Step 1 — Concept: the hidden state

- **What to do:** Write a markdown cell explaining the hidden state. At each time step, the RNN takes two inputs: the current word embedding and the previous hidden state. It produces one output: a new hidden state. The hidden state carries memory across the sequence.
- **Display in notebook:** Draw a diagram of one unrolled RNN step — input word + previous hidden state → weight matrix → tanh → new hidden state.
- **Infer from output:** The hidden state at step T theoretically encodes everything the model has seen from step 1 to T. The next task shows why "theoretically" is the key word.

---

#### Step 2 — Manual unroll across a 5-word sequence

- **What to do:** Initialize a random weight matrix and a zero hidden state. Process a 5-word sentence one word at a time: at each step multiply [word_vector; hidden_state] by the weight matrix and apply tanh. Print the hidden state after each word.
- **Display in notebook:** Print the hidden state vector (first 5 values) after processing word 1, 2, 3, 4, 5.
- **Infer from output:** Observe how the hidden state changes with each word. Now ask: how much of word 1's information survives in the hidden state after processing word 5? This motivates the gradient experiment next.

---

### Task 3 — The Vanishing Gradient Problem

**Objective:** Demonstrate why vanilla RNNs fail to learn from long-range dependencies.

---

#### Step 1 — Track gradient norms over a short sequence (length 5)

- **What to do:** Build a simple 1-layer RNN. Pass a sequence of length 5. After the backward pass, extract and print the gradient of the loss with respect to the hidden state at each time step (t=5, t=4, t=3, t=2, t=1).
- **Display in notebook:** Print the gradient norm at each time step as a table. Plot as a bar chart.
- **Infer from output:** Gradient magnitude at t=5 is largest. At t=1 it should be smaller. Note the ratio (t=5 gradient) / (t=1 gradient).

---

#### Step 2 — Repeat for sequence lengths 25 and 100

- **What to do:** Repeat the exact same experiment with sequences of length 25 and 100.
- **Display in notebook:** Plot three bar charts side by side: one for each sequence length. Use the same y-axis scale.
- **Infer from output:** For length 100, the gradient at t=1 should be near zero — often displayed as `1e-15` or smaller. This means the model learns nothing from the first word. Print the exact ratio of first/last gradients for all three experiments. This is the vanishing gradient problem.

---

### Task 4 — LSTM: Gate-by-Gate Breakdown

**Objective:** Understand what each LSTM gate does and why it solves the vanishing gradient problem.

---

#### Step 1 — The forget gate

- **What to do:** Write a markdown cell explaining the forget gate: a sigmoid function applied to [current input; previous hidden state]. Its output ranges from 0 (forget everything) to 1 (remember everything). It multiplies element-wise with the previous cell state.
- **Display in notebook:** Compute the forget gate output manually for a toy example. Print the cell state before and after applying the forget gate. Highlight which memory dimensions were reduced.
- **Infer from output:** The forget gate learns which information is irrelevant for the current context. When processing the word "but" in a sentiment sentence, the forget gate should activate strongly to reset earlier positive/negative signals.

---

#### Step 2 — The input gate and candidate values

- **What to do:** Explain: the input gate (sigmoid) controls how much of the new information to write. The candidate vector (tanh) is the new information to potentially write. The actual update is their element-wise product.
- **Display in notebook:** Compute input gate and candidate values for the same toy example. Print the element-wise product.
- **Infer from output:** The cell state update is the sum of (forget gate × old cell state) + (input gate × candidate). Print the final updated cell state and compare to the input cell state.

---

#### Step 3 — The output gate

- **What to do:** Explain: the output gate (sigmoid) controls what to expose from the cell state as the hidden state. The hidden state is (output gate × tanh(cell state)).
- **Display in notebook:** Compute the output gate and final hidden state for the same toy example. Print all gate activations and the resulting hidden state.
- **Infer from output:** Together the three gates give the LSTM full control over what to store, what to discard, and what to expose. This is what the vanilla RNN lacks — it has no mechanism to selectively preserve long-range information.

---

### Task 5 — LSTM Training for Sentiment Classification

**Objective:** Train a full LSTM model and observe its learning dynamics.

---

#### Step 1 — Prepare the data pipeline

- **What to do:** Load the IMDB dataset. Build a vocabulary from the training set. Map each word to an integer index. Pad all sequences to the same length. Create train and test dataloaders.
- **Display in notebook:** Print vocabulary size. Print 5 sample sequences in their integer-encoded form. Print the maximum and average sequence length.
- **Infer from output:** Long sequences (>500 tokens) may need truncation. The vocabulary size determines your embedding matrix size. Verify the dataset is balanced.

---

#### Step 2 — Define the model architecture

- **What to do:** Build a model with: Embedding layer → LSTM layer (bidirectional, 2 layers) → Dropout → Linear classifier head.
- **Display in notebook:** Print the model architecture summary including layer names, input/output shapes, and total trainable parameter count.
- **Infer from output:** Count the parameters in each component. The embedding layer typically dominates. Understand why bidirectional means 2× the hidden dimensions.

---

#### Step 3 — Train and monitor

- **What to do:** Train for 5–10 epochs. After every epoch, record train loss, validation loss, and validation accuracy.
- **Display in notebook:** Plot train loss and validation loss on the same graph. Plot validation accuracy over epochs separately.
- **Infer from output:** If validation loss starts increasing while training loss keeps decreasing — that is overfitting. Identify the epoch where validation loss is minimized. That is your best model.

---

#### Step 4 — Gate activation visualization

- **What to do:** After training, pass a test sentence through the model and extract the forget gate activations at each time step.
- **Display in notebook:** Plot the forget gate activations as a heatmap: x-axis is the word, y-axis is hidden units (or an averaged scalar). Color intensity represents how strongly the gate fired.
- **Infer from output:** Words that trigger high forget gate values cause the model to reset its memory — typically words like "but", "however", "although". Identify these in your test sentence.

---

### Task 6 — GRU vs LSTM Comparison

**Objective:** Understand GRU as a simplified LSTM and when to prefer it.

---

#### Step 1 — Train GRU with identical settings

- **What to do:** Replace the LSTM layer with a GRU layer. Keep all other hyperparameters identical. Train for the same number of epochs.
- **Display in notebook:** Print parameter counts for both models. Plot the training curves for both on the same axes (using different line colors).
- **Infer from output:** GRU has fewer parameters (no cell state — only hidden state). Training is faster. Record the parameter count difference and the accuracy difference. Print your conclusion: when would you choose GRU over LSTM?

---


## Phase 6 — Attention Mechanism & Transformers

**Duration:** Weeks 11–13 | **LinkedIn posts:** 4 | **Difficulty:** ⭐⭐⭐⭐⭐

### Overview

Attention solves the Seq2Seq bottleneck where the entire input sequence was compressed into a single fixed vector. With attention, the model can look at any position in the input when producing each output. The Transformer — built entirely from attention — replaced RNNs and powers every model in Phase 7 and beyond.

### Sequential Task Flow

```
Seq2Seq Bottleneck (motivation) → Attention Scores (Q×Kᵀ)
→ Scaling (÷√d_k) → Softmax (attention weights)
→ Weighted Sum (output) → Multi-Head Attention
→ Positional Encoding → Transformer Encoder Layer
→ Full Encoder Stack → Classification → LSTM Comparison
```

---

### Task 1 — Motivation: The Seq2Seq Bottleneck

**Objective:** Show the information compression problem that attention solves.

---

#### Step 1 — Visualize the bottleneck

- **What to do:** Draw and describe a Seq2Seq encoder-decoder. The encoder reads a 20-word sentence and compresses it into a single fixed-size vector (the context vector). The decoder must generate the output from only that vector.
- **Display in notebook:** Show a diagram. Then create a table: sequence length (5, 10, 20, 50 words) vs information loss (conceptual — how much context can a single 128-dim vector hold?).
- **Infer from output:** For long sequences, a single vector cannot encode everything. The decoder "forgets" early parts of the input. This is why attention was invented — instead of one fixed vector, the decoder gets to look at every encoder hidden state at each decoding step.

---

### Task 2 — Scaled Dot-Product Attention

**Objective:** Implement attention step by step.

---

#### Step 1 — Compute Q, K, V matrices

- **What to do:** Start with an input matrix of shape `(sequence_length × embedding_dim)`. Create three random weight matrices `W_Q`, `W_K`, `W_V`. Compute `Q = input × W_Q`, `K = input × W_K`, `V = input × W_V`.
- **Display in notebook:** Print the shapes of Q, K, and V. Print the first row of each matrix.
- **Infer from output:** Q, K, and V all have the same shape. They are three different linear projections of the same input. Intuitively: Q is "what am I looking for", K is "what do I have to offer", V is "what I will actually share if selected".

---

#### Step 2 — Compute raw attention scores

- **What to do:** Compute `scores = Q × K^T`. This is a matrix multiplication giving shape `(seq_len × seq_len)`.
- **Display in notebook:** Print the scores matrix for a toy sequence of length 5. Visualize it as a heatmap.
- **Infer from output:** Each cell `[i, j]` represents how much position `i` should attend to position `j`. Higher = more attention. But the scores are unbounded — they need scaling and normalization.

---

#### Step 3 — Apply scaling

- **What to do:** Divide the scores matrix by the square root of the key dimension (`√d_k`).
- **Display in notebook:** Print the scores before and after scaling. Compare the minimum and maximum values.
- **Infer from output:** Scaling prevents scores from becoming extremely large in high dimensions. Large scores cause softmax to produce near-zero gradients — the model stops learning. The `√d_k` factor keeps scores in a range where softmax gradients are non-trivial.

---

#### Step 4 — Apply softmax row-wise

- **What to do:** Apply softmax independently to each row of the scaled scores matrix.
- **Display in notebook:** Print the resulting attention weight matrix. Verify that each row sums to exactly 1.0. Visualize as a heatmap.
- **Infer from output:** The attention weight matrix is now a probability distribution per query position. Row `i` shows what percentage of attention position `i` pays to each other position. Look for which positions attend strongly to which others.

---

#### Step 5 — Compute the weighted sum output

- **What to do:** Multiply the attention weight matrix by `V` (the values matrix).
- **Display in notebook:** Print the output shape. Print the first output vector and compare it to the first V vector.
- **Infer from output:** The output at each position is a weighted average of all value vectors, weighted by attention. Positions with high attention weight contribute more to the output. This is the core operation of the entire transformer.

---

### Task 3 — Multi-Head Attention

**Objective:** Run attention in parallel across multiple learned subspaces.

---

#### Step 1 — Split into multiple heads

- **What to do:** Take the embedding dimension (e.g., 64) and split it into 4 heads of dimension 16 each. Create separate `W_Q`, `W_K`, `W_V` for each head.
- **Display in notebook:** Print the shapes of Q, K, V for one head. Print all 4 heads' Q shapes to confirm they are identical.
- **Infer from output:** Each head attends to a different linear projection of the input. Different heads can learn to detect different relationship types — syntactic, semantic, positional.

---

#### Step 2 — Compute attention per head and concatenate

- **What to do:** Run scaled dot-product attention separately for all 4 heads. Concatenate the outputs along the last dimension. Project back to the original dimension with `W_O`.
- **Display in notebook:** Print the shapes at each step: per-head output, concatenated output, final projected output. Confirm the final shape matches the input shape.
- **Infer from output:** The final output has the same shape as the input — multi-head attention is a shape-preserving transformation that enriches each position's representation with global context.

---

### Task 4 — Positional Encoding

**Objective:** Inject position information since transformers have no built-in sequence order.

---

#### Step 1 — Demonstrate that transformers are order-blind without PE

- **What to do:** Take the same set of words and pass them through your attention module in two different orders. Compare the output at position 1 in both cases.
- **Display in notebook:** Print both outputs. Check whether they differ.
- **Infer from output:** Without positional encoding, the output at position 1 is the same regardless of word order (the attention weights will differ but only because the input vectors differ — not because of position). The model has no way to know that word at position 3 comes before word at position 7.

---

#### Step 2 — Compute sinusoidal positional encoding

- **What to do:** For each position (0 to max_len) and each dimension (0 to d_model), compute the sinusoidal encoding: sine for even dimensions, cosine for odd dimensions. The frequency decreases with dimension index.
- **Display in notebook:** Print the positional encoding matrix as a table for the first 10 positions and first 10 dimensions. Visualize the full matrix as a heatmap.
- **Infer from output:** Each position gets a unique "fingerprint" vector. Nearby positions have similar vectors (because the sinusoids change slowly at low frequencies). This encodes relative position naturally.

---

#### Step 3 — Add positional encoding to word embeddings

- **What to do:** Add the positional encoding matrix to the word embedding matrix element-wise.
- **Display in notebook:** Print a word's embedding vector before and after adding positional encoding. Compute the cosine similarity between the same word's embedding at position 1 vs position 10.
- **Infer from output:** The same word now has a different vector depending on its position. Position 1 and position 10 versions of the same word are similar (high cosine similarity) but not identical — the positional signal is embedded.

---

### Task 5 — Full Transformer Encoder Layer

**Objective:** Stack attention, normalization, and feed-forward into one complete encoder layer.

---

#### Step 1 — Build and describe the encoder layer components

- **What to do:** Assemble: (1) Multi-Head Self-Attention, (2) Add & Norm (residual connection + layer norm), (3) Feed-Forward Network (two linear layers with ReLU), (4) Add & Norm again. This is one encoder layer.
- **Display in notebook:** Print the complete architecture summary including every sublayer, its input/output shape, and its parameter count.
- **Infer from output:** Identify the residual connections. These skip connections ensure gradients can flow directly to early layers without passing through the attention sublayer — this prevents the vanishing gradient problem at depth.

---

#### Step 2 — Train encoder on a classification task

- **What to do:** Stack 2 encoder layers. Add a mean-pooling step followed by a linear classification head. Train on a text classification dataset (use the same dataset as Phase 5 LSTM).
- **Display in notebook:** Plot training and validation curves. Print the final classification report.
- **Infer from output:** Compare these curves to the LSTM curves from Phase 5. Does the transformer converge faster or slower? Does it achieve higher accuracy? Print both models' final test accuracy side by side.

---

#### Step 3 — Visualize attention weights

- **What to do:** After training, pass a test sentence through the encoder and extract the attention weight matrix from the first layer. Visualize as a heatmap with word labels.
- **Display in notebook:** Display the heatmap with words on both axes. Highlight cells with weight > 0.1.
- **Infer from output:** Look for which words the model attends to most strongly. Are there interpretable patterns? Does the word "not" attend to the next word (as expected for negation handling)?

---


## Phase 7 — BERT, GPT & Pre-trained Models

**Duration:** Weeks 14–17 | **LinkedIn posts:** 4–5 | **Difficulty:** ⭐⭐⭐⭐⭐

### Overview

BERT and GPT are transformers pre-trained on massive text corpora. The key insight: instead of training a model from scratch on your task (which requires large labelled datasets), you start with a model that already understands language deeply, then fine-tune it for your specific task with a small labelled dataset.

### Sequential Task Flow

```
BERT Tokenizer Inspection → BERT Architecture Understanding
→ Fine-tuning for Classification → Evaluation & Interpretation
→ GPT-2 Text Generation → Zero-Shot Classification
→ Few-Shot Prompting → Model Size Comparison
```

---

### Task 1 — Understanding BERT's Tokenizer

**Objective:** See how BERT processes text before it enters the model.

---

#### Step 1 — Load and inspect the BERT tokenizer

- **What to do:** Load `bert-base-uncased` tokenizer. Tokenize 5 sample sentences. Print the raw tokens, input IDs, attention mask, and token type IDs for each.
- **Display in notebook:** Print all four outputs in aligned columns. Note the `[CLS]` token at position 0 and `[SEP]` at the end.
- **Infer from output:** `[CLS]` is a special token whose final hidden state is used as the sentence-level representation for classification. `[SEP]` separates sentences in tasks involving sentence pairs. The attention mask is 1 for real tokens and 0 for padding — the model ignores padded positions.

---

#### Step 2 — Observe WordPiece subword splitting

- **What to do:** Tokenize sentences containing rare words, technical terms, and deliberately misspelled words.
- **Display in notebook:** Print the tokenized output for each. Highlight any token that starts with `##` (indicating a subword continuation).
- **Infer from output:** Common words are one token. Rare or long words are split into multiple subword tokens — for example, `"tokenization"` → `["token", "##ization"]`. This is how BERT handles words it never saw during pre-training: it decomposes them into known subword units.

---

### Task 2 — BERT Architecture and Pre-training Objectives

**Objective:** Understand what BERT learned during pre-training.

---

#### Step 1 — Masked Language Modeling (MLM)

- **What to do:** Use the `fill-mask` pipeline to test BERT's masked language model. Mask different words in 5 sentences and observe BERT's top-5 predictions with confidence scores.
- **Display in notebook:** Print the sentence with the `[MASK]`, then print the top-5 predicted words and their probabilities.
- **Infer from output:** BERT predicts masked words using context from both sides simultaneously (bidirectional). Compare its predictions for a masked noun vs a masked adjective. Does it predict grammatically appropriate words? This is what 340M pre-training steps produced.

---

#### Step 2 — Extract CLS token embeddings

- **What to do:** Pass 10 sentences through BERT (without the classification head). Extract the `[CLS]` token's hidden state vector from the final layer for each sentence.
- **Display in notebook:** Print the shape of each CLS vector. Compute pairwise cosine similarity between all 10 CLS vectors. Visualize as a heatmap.
- **Infer from output:** Sentences with similar meaning should have high CLS cosine similarity. Verify this against your intuition for at least 3 pairs. This embedding can be used as a sentence representation for downstream tasks.

---

### Task 3 — Fine-tuning BERT for Classification

**Objective:** Adapt a pre-trained BERT to a specific text classification task.

---

#### Step 1 — Prepare and tokenize the dataset

- **What to do:** Load your chosen classification dataset. Apply the BERT tokenizer with truncation and padding to a max length of 128 tokens. Inspect the tokenized batch.
- **Display in notebook:** Print a sample tokenized example showing input IDs, attention mask, and true label. Print the distribution of sequence lengths as a histogram.
- **Infer from output:** If most sequences are much shorter than 128 tokens, you can reduce `max_length` to speed up training. The histogram tells you the right truncation cutoff.

---

#### Step 2 — Define and inspect the model

- **What to do:** Load `AutoModelForSequenceClassification` with `num_labels` set to your task's class count. Print the model architecture, focusing on the classifier head that was randomly initialized.
- **Display in notebook:** Print total parameters. Print trainable parameters. Print only the classifier head parameters separately.
- **Infer from output:** The pre-trained transformer layers have millions of parameters already initialized with language knowledge. The randomly initialized head has only a few hundred or thousand parameters. Fine-tuning updates all of them together, but the transformer layers need very small learning rate updates.

---

#### Step 3 — Train and monitor

- **What to do:** Train for 3 epochs using `Trainer` API. Log train loss, validation loss, and validation accuracy at each epoch.
- **Display in notebook:** Plot training and validation loss curves. Print the best validation accuracy achieved and at which epoch.
- **Infer from output:** BERT typically converges in 2–3 epochs on classification tasks. If validation loss plateaus or increases after epoch 1, you may be overfitting — BERT is very powerful and can memorize a small dataset quickly.

---

#### Step 4 — Evaluate and analyse errors

- **What to do:** Generate predictions on the full test set. Print the classification report. Plot the confusion matrix.
- **Display in notebook:** Show the classification report and confusion matrix.
- **Infer from output:** Identify the class with the worst F1 score. Pull out 10 examples from that class that the model got wrong. Read them carefully — are they ambiguous? Are they a specific writing style? This error analysis guides what data to collect next.

---

### Task 4 — GPT-2 Text Generation

**Objective:** Understand autoregressive (left-to-right) generation and its controls.

---

#### Step 1 — Greedy decoding

- **What to do:** Load `GPT2LMHeadModel`. Generate text from a prompt using greedy decoding (always pick the highest probability next token).
- **Display in notebook:** Print the generated text.
- **Infer from output:** Greedy output is often repetitive or bland — once a common phrase starts, the greedy strategy gets trapped in loops. Note any repeated phrases.

---

#### Step 2 — Temperature sampling

- **What to do:** Generate from the same prompt 5 times using temperature = 0.3, 0.7, 1.0, 1.5, 2.0.
- **Display in notebook:** Print all 5 generated texts with their temperature label.
- **Infer from output:** Temperature 0.3 is conservative and coherent but repetitive. Temperature 2.0 is creative but often nonsensical. Temperature 0.7–1.0 is the typical sweet spot. Record which temperature you subjectively prefer and why.

---

#### Step 3 — Top-k and nucleus (top-p) sampling

- **What to do:** Generate with `top_k=50` (only sample from the top 50 tokens) and with `top_p=0.9` (sample from tokens that collectively account for 90% probability mass).
- **Display in notebook:** Print outputs for each setting. Compare to the temperature sampling outputs.
- **Infer from output:** Nucleus sampling adapts the vocabulary pool per step — when the model is confident it restricts options; when uncertain it broadens them. This produces more natural variation than fixed top-k.

---

### Task 5 — Zero-Shot and Few-Shot Inference

**Objective:** Use pre-trained models for tasks they were never explicitly trained on.

---

#### Step 1 — Zero-shot classification

- **What to do:** Use the `zero-shot-classification` pipeline with `facebook/bart-large-mnli`. Provide a sentence and 4–5 candidate labels. Print the scores.
- **Display in notebook:** Print the sentence, each candidate label, and its confidence score. Sort by score descending.
- **Infer from output:** The model was never trained on your specific labels. Yet it assigns meaningful probabilities. This works because NLI training (does hypothesis A follow from premise B?) generalises to classification. How close is the top score to 1.0? For unambiguous sentences, it should be high.

---

#### Step 2 — Few-shot prompting comparison

- **What to do:** Test the same 10 classification queries with three prompt styles: zero-shot (just the query), one-shot (one example before the query), three-shot (three examples before the query).
- **Display in notebook:** Print a table: query | zero-shot prediction | one-shot prediction | three-shot prediction.
- **Infer from output:** Count how many predictions change between zero-shot, one-shot, and three-shot. For which query types do examples help most? This is your first structured prompt engineering experiment.

---


## Phase 8 — RAG, Prompting & Fine-tuning

**Duration:** Weeks 18–22 | **LinkedIn posts:** 5–6 | **Difficulty:** ⭐⭐⭐⭐⭐

### Overview

The final phase covers production NLP. RAG connects LLMs to private knowledge without retraining. Prompt engineering extracts maximum performance by crafting inputs carefully. LoRA fine-tuning adapts billion-parameter models by training less than 1% of their weights. This phase answers: how do you build real NLP systems?

### Sequential Task Flow

```
Semantic Search (embedding → FAISS) → RAG Pipeline (chunk → embed → index → retrieve → generate)
→ Prompt Engineering (zero-shot → few-shot → CoT → role)
→ LoRA Fine-tuning (config → train → compare)
→ Evaluation (BLEU → ROUGE → BERTScore → comparison)
```

---

### Task 1 — Semantic Search

**Objective:** Build a search system that understands meaning, not just keywords.

---

#### Step 1 — Embed queries and documents

- **What to do:** Load a `SentenceTransformer` model (`all-MiniLM-L6-v2`). Embed a collection of 20 short documents and 5 test queries.
- **Display in notebook:** Print the shape of the document embedding matrix. Print the vector for one document (first 10 values).
- **Infer from output:** Every document is now a fixed-size dense vector regardless of its length. The semantic meaning is compressed into this vector. Documents about the same topic should be near each other in this space.

---

#### Step 2 — Compute semantic similarity

- **What to do:** Compute cosine similarity between each query vector and all document vectors. For each query, rank documents by similarity.
- **Display in notebook:** For each query, print the top-3 most similar documents with their similarity scores.
- **Infer from output:** Verify that the retrieved documents are semantically related to the query — not just sharing keywords. Test a paraphrase query (different words, same meaning) and verify the correct document is still retrieved.

---

#### Step 3 — Compare semantic search vs keyword search (TF-IDF)

- **What to do:** Run the same 5 queries through TF-IDF cosine similarity. Compare the ranked results to semantic search.
- **Display in notebook:** Print a side-by-side comparison table: query | TF-IDF top-1 result | Semantic top-1 result.
- **Infer from output:** Find at least one query where semantic search retrieves a clearly better document. This is typically a paraphrase query where the query and document share no exact words. This is the fundamental advantage of embedding-based search.

---

### Task 2 — Build a Full RAG Pipeline

**Objective:** Combine retrieval and generation to answer questions grounded in a knowledge base.

---

#### Step 1 — Chunk your knowledge base

- **What to do:** Take a collection of documents (your NLP notes from Phases 1–7 work perfectly). Split each document into chunks of ~300 words with 50-word overlap between consecutive chunks.
- **Display in notebook:** Print the first 3 chunks with their source document label. Print the total number of chunks created.
- **Infer from output:** Smaller chunks improve retrieval precision but lose cross-sentence context. Larger chunks preserve context but reduce precision. The 50-word overlap ensures sentences at chunk boundaries are not lost.

---

#### Step 2 — Embed all chunks and build FAISS index

- **What to do:** Embed all chunks using the SentenceTransformer. Build a FAISS IndexFlatIP (inner product index) from the embeddings. Save the index to disk.
- **Display in notebook:** Print the total number of vectors in the index. Print the index dimensionality.
- **Infer from output:** The FAISS index enables sub-millisecond nearest-neighbour search even across millions of vectors. This is what makes RAG practical at scale.

---

#### Step 3 — Retrieve relevant chunks for a query

- **What to do:** Take a test question. Embed it using the same SentenceTransformer. Search the FAISS index for the top 3 most similar chunks.
- **Display in notebook:** Print the query. Print the 3 retrieved chunks with their similarity scores and source labels.
- **Infer from output:** Are the retrieved chunks relevant to the query? If not, investigate why — is the query too vague? Is the correct information in the knowledge base? Retrieval quality is the most important determinant of RAG system quality.

---

#### Step 4 — Construct the augmented prompt

- **What to do:** Format a prompt that includes: an instruction ("Answer the question using only the provided context"), the 3 retrieved chunks as numbered context passages, and the user's question.
- **Display in notebook:** Print the full formatted prompt. Count the total token length.
- **Infer from output:** The prompt must fit within the model's context window. For long retrieved chunks, you may need to truncate. Observe how the context is formatted — this formatting matters for the model's ability to use it.

---

#### Step 5 — Generate the answer

- **What to do:** Pass the augmented prompt to a language model. Print the generated answer.
- **Display in notebook:** Print the full prompt and the generated answer side by side.
- **Infer from output:** Does the answer correctly use the retrieved context? Does it hallucinate information not in the context? Test 10 questions and categorize each: (1) correct and grounded, (2) correct but not grounded, (3) incorrect. The failure modes reveal system weaknesses.

---

### Task 3 — Prompt Engineering

**Objective:** Systematically test prompt strategies and measure their effect on output quality.

---

#### Step 1 — Zero-shot prompting

- **What to do:** Construct a minimal prompt: task instruction + input, no examples.
- **Display in notebook:** Print the prompt and the model output for 10 test queries.
- **Infer from output:** Identify queries where the model clearly misunderstands the task. These become your test cases for improving the prompt.

---

#### Step 2 — Few-shot prompting

- **What to do:** Add 3 example (input, output) pairs before each query. Keep the examples diverse and representative of the task.
- **Display in notebook:** Print the few-shot prompt for one query. Print outputs for all 10 test queries. Highlight cases that changed from zero-shot.
- **Infer from output:** Count how many of the 10 queries improved, stayed the same, or got worse. Record the overall accuracy change.

---

#### Step 3 — Chain-of-Thought prompting

- **What to do:** Add "Let's think step by step" or include a worked-out reasoning chain in your examples. Apply this to 5 multi-step reasoning questions.
- **Display in notebook:** Print the full chain-of-thought output for each question. Show whether the final answer improved compared to direct prompting.
- **Infer from output:** Chain-of-thought helps on tasks that require multi-step logic. It typically hurts on tasks that need fast, direct recall. Identify which of your 5 questions benefited from CoT and which did not.

---

#### Step 4 — Role prompting

- **What to do:** Prefix your prompt with "You are an expert NLP researcher..." or a domain-appropriate persona.
- **Display in notebook:** Show output for the same query with and without the role prefix.
- **Infer from output:** Role prompting often improves terminology accuracy and output structure. Note whether the improvement is in style (more professional language) or in substance (more accurate content).

---

#### Step 5 — Ablation summary table

- **What to do:** Collect all your prompt strategy results. Build a summary table: strategy | accuracy | avg response length | qualitative notes.
- **Display in notebook:** Print the full comparison table. Plot a bar chart of accuracy by strategy.
- **Infer from output:** Write a 5-line conclusion: which strategy worked best overall, which worked best for which task type, and which had unexpected negative effects. This becomes your personal prompt engineering guide.

---

### Task 4 — LoRA Fine-tuning

**Objective:** Fine-tune a large language model efficiently by training only a tiny fraction of parameters.

---

#### Step 1 — Understand LoRA conceptually

- **What to do:** Write a markdown cell explaining LoRA. A weight matrix `W` (large) is approximated by freezing `W` and adding a low-rank product `A × B` where `A` and `B` are much smaller matrices. Only `A` and `B` are trained.
- **Display in notebook:** Draw a diagram showing the original weight matrix, the two LoRA matrices, and how they combine during the forward pass.
- **Infer from output:** If `W` is 4096×4096 and rank `r=8`, then `A` is 4096×8 and `B` is 8×4096. Total parameters in `A+B`: 65,536. Parameters in `W`: 16,777,216. LoRA adds 0.39% extra parameters. Print these exact numbers.

---

#### Step 2 — Apply LoRA configuration and inspect trainable parameters

- **What to do:** Load a small base model (125M–1B parameters). Apply `LoraConfig` targeting the query and value projection matrices with `r=8`, `lora_alpha=32`. Call `print_trainable_parameters()`.
- **Display in notebook:** Print the number of trainable parameters, total parameters, and the percentage. Print which specific layer names have LoRA adapters attached.
- **Infer from output:** Confirm the trainable parameter count is under 1% of total. Print a clear comparison: "Before LoRA: X trainable params. After LoRA: Y trainable params (Z% of total)."

---

#### Step 3 — Prepare the instruction-tuning dataset

- **What to do:** Format your dataset as instruction-response pairs. Print 5 examples in the exact format the model will see during training: system instruction + user input + assistant response.
- **Display in notebook:** Print the 5 formatted examples. Print the dataset size (train/validation split).
- **Infer from output:** The formatting must be consistent. A single malformatted example can degrade the whole fine-tune. Verify every example follows the same template.

---

#### Step 4 — Train and monitor loss

- **What to do:** Train for 1–3 epochs using `SFTTrainer`. Log training loss at every 10 steps.
- **Display in notebook:** Plot the training loss curve. Mark the loss at the start and end of training.
- **Infer from output:** Loss should decrease monotonically. If it plateaus early, the learning rate may be too low. If it oscillates wildly, it may be too high. Note the final loss value.

---

#### Step 5 — Before vs after comparison

- **What to do:** Generate outputs for 10 test prompts from the base model (before fine-tuning) and the LoRA-adapted model (after fine-tuning). Print both outputs side by side.
- **Display in notebook:** Print the 10 pairs in a before/after format. Highlight the most striking improvements.
- **Infer from output:** Categorize the changes: Did the model's output format improve? Did it adopt domain-specific vocabulary? Did it become more concise or more detailed? This qualitative analysis is as important as any metric.

---

### Task 5 — Evaluation Metrics

**Objective:** Measure generated text quality quantitatively and understand what each metric misses.

---

#### Step 1 — BLEU score

- **What to do:** Compute BLEU-1, BLEU-2, BLEU-3, BLEU-4 for 10 (generated, reference) pairs using the `evaluate` library.
- **Display in notebook:** Print the score for each test pair and the mean across all 10. Print one example where BLEU is high and one where it is low.
- **Infer from output:** High BLEU means strong n-gram overlap. Low BLEU can mean a semantically correct answer phrased differently from the reference. Print a case where the generated text is clearly correct but BLEU is low — this shows BLEU's limitation.

---

#### Step 2 — ROUGE score

- **What to do:** Compute ROUGE-1, ROUGE-2, and ROUGE-L for the same 10 pairs.
- **Display in notebook:** Print a table: pair ID | ROUGE-1 | ROUGE-2 | ROUGE-L.
- **Infer from output:** ROUGE-1 and ROUGE-2 measure unigram and bigram recall. ROUGE-L measures the longest common subsequence. A high ROUGE-L with low ROUGE-2 means the generated text preserves the broad sequence but not exact phrase matches.

---

#### Step 3 — BERTScore

- **What to do:** Compute BERTScore precision, recall, and F1 for the same 10 pairs.
- **Display in notebook:** Print P, R, F1 for each pair. Print the mean F1.
- **Infer from output:** Find a case where BLEU is low but BERTScore F1 is high. This is a paraphrase — same meaning, different words. BERTScore captures semantic similarity that BLEU misses entirely.

---

#### Step 4 — Metrics comparison summary

- **What to do:** Build a final summary table with all three metrics for all 10 pairs. Sort by BERTScore F1.
- **Display in notebook:** Print the full table. Plot a bar chart comparing the three metrics' mean scores across all pairs.
- **Infer from output:** Write a 3-line conclusion for each metric: what it measures, what it misses, and the task type it is best suited for. This becomes your reference guide for evaluating NLP systems.

---


## LinkedIn Content Strategy

### The Post Formula That Works

Every post should follow this five-part structure:

1. **Hook** — a surprising number, a counterintuitive result, or a specific failure that creates curiosity
2. **Context** — what you were trying to do (1–2 sentences)
3. **Insight** — the concrete thing you learned
4. **Visual** — a screenshot of your notebook output, a plot, a heatmap, or a before/after table
5. **Question** — one specific question that invites your audience to engage

### Posting Cadence

| Phase Range                | Total Posts | Frequency       |
| -------------------------- | ----------- | --------------- |
| Phases 1–3 (Weeks 1–4)   | 7–8 posts  | Every 3–4 days |
| Phases 4–5 (Weeks 5–10)  | 7–8 posts  | Every 3–4 days |
| Phases 6–7 (Weeks 11–17) | 8–9 posts  | Every 3 days    |
| Phase 8 (Weeks 18–22)     | 5–6 posts  | Every 3–5 days |

### Anchor Posts

- **Launch (Week 1):** "I am starting a public NLP learning journey — from tokenization to fine-tuning transformers, every step documented here. Follow along. Post 1 of ~30."
- **Milestone (After Phase 3):** "I just trained Word2Vec from scratch and visualized the embedding space. Here is what the t-SNE cluster looks like — and what king − man + woman equals on my data."
- **Midpoint (After Phase 6):** "I just implemented transformer attention line by line. Here are the 3 things I did not understand until I coded them myself."
- **Completion (After Phase 8):** "22 weeks. 8 phases. 30+ posts. Here is every concept I built, every notebook I wrote, and the GitHub repo. NLP from scratch — done."

Use `#NLPFromScratch` on every single post to build a discoverable series.

---

## Portfolio & GitHub Structure

### Repository: `nlp-from-scratch`

```
nlp-from-scratch/
├── README.md
├── phase-01-tokenization/
│   ├── README.md
│   ├── 01_data_cleaning.ipynb
│   ├── 02_word_tokenization.ipynb
│   ├── 03_sentence_tokenization.ipynb
│   ├── 04_stopword_removal.ipynb
│   ├── 05_stemming_lemmatization.ipynb
│   ├── 06_full_pipeline.ipynb
│   └── outputs/
├── phase-02-bow-tfidf/
│   ├── 01_one_hot_encoding.ipynb
│   ├── 02_bag_of_words.ipynb
│   ├── 03_ngrams.ipynb
│   ├── 04_tfidf.ipynb
│   └── 05_cosine_similarity.ipynb
├── phase-03-word-embeddings/
├── phase-04-classical-nlp/
├── phase-05-rnn-lstm/
├── phase-06-attention-transformers/
├── phase-07-bert-gpt/
├── phase-08-rag-finetuning/
└── datasets/
```

### Phase README Template

Each phase folder's `README.md` answers five questions:

1. What problem does this phase solve?
2. What are the key concepts? (one sentence each)
3. How do I run the notebooks? (order + install commands)
4. What did each exercise reveal? (updated after completion)
5. Where can I learn more? (one paper + one tutorial + LinkedIn link)

---

## Resources & References

### Papers to Read (by phase)

| Phase   | Paper                                                                                      |
| ------- | ------------------------------------------------------------------------------------------ |
| Phase 3 | Mikolov et al. (2013) —*Efficient Estimation of Word Representations in Vector Space*   |
| Phase 5 | Hochreiter & Schmidhuber (1997) —*Long Short-Term Memory*                               |
| Phase 6 | Vaswani et al. (2017) —*Attention Is All You Need*                                      |
| Phase 7 | Devlin et al. (2018) —*BERT: Pre-training of Deep Bidirectional Transformers*           |
| Phase 7 | Radford et al. (2019) —*Language Models are Unsupervised Multitask Learners* (GPT-2)    |
| Phase 8 | Lewis et al. (2020) —*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* |
| Phase 8 | Hu et al. (2021) —*LoRA: Low-Rank Adaptation of Large Language Models*                  |

### Datasets by Phase

| Phase   | Dataset             | Task                    |
| ------- | ------------------- | ----------------------- |
| Phase 2 | 20 Newsgroups       | TF-IDF + classification |
| Phase 4 | SMS Spam Collection | Text classification     |
| Phase 4 | IMDB Reviews        | Sentiment analysis      |
| Phase 4 | CoNLL-2003          | NER                     |
| Phase 5 | IMDB Reviews        | Sequence classification |
| Phase 5 | Multi30k            | Seq2Seq                 |
| Phase 7 | IMDB / AG News      | BERT fine-tuning        |
| Phase 8 | Alpaca              | LoRA instruction tuning |

### Essential Reading

- **The Illustrated Transformer** — Jay Alammar (best visual explanation of transformers)
- **The Illustrated BERT** — Jay Alammar
- **Lilian Weng's blog** (lilianweng.github.io) — deep write-ups on attention and RAG
- **Hugging Face NLP Course** (huggingface.co/learn/nlp-course) — free, hands-on

---

*Last updated: June 2026 · Built as a public learning journey from tokenization to RAG*
