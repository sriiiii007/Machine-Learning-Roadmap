# NLP - Complete Theory Guide 📚

Comprehensive theoretical understanding of Natural Language Processing.

---

## Table of Contents

1. [Text Preprocessing](#text-preprocessing)
2. [Text Representations](#text-representations)
3. [Classical NLP Tasks](#classical-nlp-tasks)
4. [Word Embeddings](#word-embeddings)
5. [Advanced NLP](#advanced-nlp)

---

## Text Preprocessing

### Why Preprocess?

- **Standardization**: Consistent format
- **Noise Removal**: Remove irrelevant information
- **Normalization**: Reduce vocabulary size
- **Feature Extraction**: Prepare for ML models

### Steps

1. **Lowercasing**: Convert to lowercase
2. **Tokenization**: Split into words/tokens
3. **Removing Stop Words**: Remove common words
4. **Stemming/Lemmatization**: Reduce to root form
5. **Removing Special Characters**: Clean text
6. **Handling Numbers**: Remove or normalize

### Tokenization

**Word Tokenization**: Split into words
- Simple: Split by spaces
- Advanced: Handle punctuation, contractions

**Sentence Tokenization**: Split into sentences
- Based on punctuation
- Handle abbreviations

### Stemming vs Lemmatization

**Stemming**:
- Cuts off word endings
- Fast but can produce invalid words
- Example: "running" → "run", "better" → "bet"

**Lemmatization**:
- Returns valid dictionary words
- Slower but more accurate
- Example: "running" → "run", "better" → "good"

---

## Text Representations

### Bag of Words (BoW)

**How it works**:
- Count word occurrences
- Create vocabulary
- Represent documents as word counts

**Example**:
```
Doc 1: "I love ML"
Doc 2: "ML is great"

Vocabulary: {I: 0, love: 1, ML: 2, is: 3, great: 4}

Doc 1: [1, 1, 1, 0, 0]
Doc 2: [0, 0, 1, 1, 1]
```

**Pros**:
- Simple and fast
- Easy to understand

**Cons**:
- Loses word order
- High dimensionality
- Sparse representation
- No semantics

### TF-IDF (Term Frequency-Inverse Document Frequency)

**TF (Term Frequency)**:
```
TF(t, d) = (Number of times t appears in d) / (Total words in d)
```

**IDF (Inverse Document Frequency)**:
```
IDF(t) = log(Total documents / Documents containing t)
```

**TF-IDF**:
```
TF-IDF(t, d) = TF(t, d) * IDF(t)
```

**How it works**:
- Weights words by importance
- Rare words get higher weights
- Common words get lower weights

**Pros**:
- Better than BoW
- Weights important words
- Reduces impact of common words

**Cons**:
- Still loses word order
- High dimensionality
- No semantics

### Word Embeddings

**Definition**: Dense vector representations of words

**Properties**:
- Fixed size (typically 100-300 dimensions)
- Dense (not sparse)
- Captures semantic relationships
- Similar words have similar vectors

**Types**:

1. **Word2Vec**:
   - Skip-gram or CBOW
   - Trained on large corpus
   - Captures word relationships

2. **GloVe**:
   - Global vectors
   - Based on co-occurrence matrix
   - Good performance

3. **FastText**:
   - Handles subwords
   - Good for rare words
   - Handles out-of-vocabulary words

**Advantages**:
- Captures semantics
- Lower dimensionality
- Can find similar words
- Transfer learning possible

**Disadvantages**:
- Requires training or pre-trained models
- Fixed representation (no context)
- Can't handle polysemy

### Contextual Embeddings

**Definition**: Embeddings that change based on context

**Examples**:
- BERT
- GPT
- ELMo

**Advantages**:
- Captures context
- Handles polysemy
- State-of-the-art performance

**Disadvantages**:
- Computationally expensive
- Requires large models
- Less interpretable

---

## Classical NLP Tasks

### Sentiment Analysis

**Goal**: Classify text sentiment (positive/negative/neutral)

**Approaches**:
- Rule-based (lexicon)
- Machine Learning (BoW, TF-IDF)
- Deep Learning (LSTM, BERT)

**Applications**:
- Product reviews
- Social media monitoring
- Customer feedback

### Named Entity Recognition (NER)

**Goal**: Identify and classify named entities

**Entities**:
- Person names
- Locations
- Organizations
- Dates
- Money

**Approaches**:
- Rule-based
- Conditional Random Fields (CRF)
- Deep Learning (BiLSTM, BERT)

### Part-of-Speech Tagging

**Goal**: Tag each word with its grammatical role

**Tags**:
- Noun (NN)
- Verb (VB)
- Adjective (JJ)
- etc.

**Applications**:
- Grammar checking
- Information extraction
- Machine translation

### Text Classification

**Goal**: Classify documents into categories

**Types**:
- Binary classification
- Multi-class classification
- Multi-label classification

**Applications**:
- Spam detection
- Topic classification
- Language detection

### Text Summarization

**Types**:
- **Extractive**: Select important sentences
- **Abstractive**: Generate new summary

**Approaches**:
- Statistical methods
- Graph-based
- Deep Learning (Seq2Seq, Transformers)

---

## Word Embeddings

### Word2Vec

**Skip-gram**:
- Predicts context words from target word
- Good for rare words
- More training time

**CBOW (Continuous Bag of Words)**:
- Predicts target word from context
- Faster training
- Better for frequent words

**Training**:
- Large corpus needed
- Neural network architecture
- Learns word relationships

### Properties of Word Embeddings

**Semantic Relationships**:
```
king - man + woman ≈ queen
```

**Similarity**:
- Cosine similarity between vectors
- Similar words have similar vectors

**Visualization**:
- t-SNE or PCA for 2D visualization
- Clusters of related words

### Using Pre-trained Embeddings

**Advantages**:
- No training needed
- Trained on large corpora
- Good performance out of the box

**Popular Models**:
- Word2Vec (Google News)
- GloVe (Wikipedia, Common Crawl)
- FastText (Wikipedia, News)

---

## Advanced NLP

### Transformers

**Key Innovation**: Attention mechanism

**Architecture**:
- Encoder-Decoder
- Self-attention
- Position encoding

**Models**:
- BERT (Bidirectional)
- GPT (Autoregressive)
- T5 (Text-to-Text)

### BERT (Bidirectional Encoder Representations from Transformers)

**Key Features**:
- Bidirectional context
- Masked language modeling
- Pre-trained on large corpus

**Applications**:
- Text classification
- Question answering
- Named entity recognition
- Sentiment analysis

### Fine-tuning

**Process**:
1. Start with pre-trained model
2. Add task-specific layers
3. Fine-tune on your data
4. Much faster than training from scratch

**Benefits**:
- Better performance
- Less data needed
- Faster training

---

## Best Practices

1. **Preprocess Properly**: Clean and normalize text
2. **Choose Right Representation**: Based on task
3. **Use Pre-trained Models**: When possible
4. **Handle Imbalanced Data**: For classification
5. **Evaluate Properly**: Use appropriate metrics
6. **Consider Context**: For ambiguous words
7. **Domain Adaptation**: Fine-tune for your domain

---

## Common Challenges

1. **Ambiguity**: Words with multiple meanings
2. **Sarcasm**: Hard to detect
3. **Context**: Requires understanding context
4. **Languages**: Different languages, different approaches
5. **Domain**: Medical, legal, technical jargon
6. **Data Quality**: Noisy, incomplete data

---

## Evaluation Metrics

### For Classification

- **Accuracy**: Overall correctness
- **Precision**: Of predicted positives, how many correct
- **Recall**: Of actual positives, how many found
- **F1-Score**: Balance of precision and recall

### For NER

- **Precision**: Correct entities / Predicted entities
- **Recall**: Correct entities / Actual entities
- **F1-Score**: Harmonic mean

### For Summarization

- **ROUGE**: Overlap with reference summary
- **BLEU**: Similarity to reference
- **Human Evaluation**: Best but expensive

---

## Next Steps

After mastering NLP basics:

1. ✅ Explore transformers
2. ✅ Learn about LLMs
3. ✅ Build RAG systems
4. ✅ Work with real-world applications
5. ✅ Stay updated with latest research

---

**Remember**: NLP is rapidly evolving. Master the fundamentals, then keep learning new techniques!

