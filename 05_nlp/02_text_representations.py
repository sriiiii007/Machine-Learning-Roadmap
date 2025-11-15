"""
Text Representations - From Bag of Words to Word Embeddings
Understanding different ways to represent text for ML.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import Word2Vec, FastText
from gensim.downloader import load as gensim_load
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

print("=" * 70)
print("TEXT REPRESENTATIONS - Theory & Practice")
print("=" * 70)

# ============================================================================
# THEORY: Text Representations
# ============================================================================
print("\n" + "=" * 70)
print("THEORY: Understanding Text Representations")
print("=" * 70)

theory = """
TEXT REPRESENTATIONS FOR ML

1. THE CHALLENGE:
   - ML models need numbers, not text
   - Need to convert text to numerical vectors
   - Different representations capture different information

2. EVOLUTION OF REPRESENTATIONS:

   a) BAG OF WORDS (BoW)
      - Count word occurrences
      - Simple but loses word order
      - High-dimensional, sparse
      - Example: "I love ML" → [1, 1, 1, 0, 0, ...]
   
   b) TF-IDF (Term Frequency-Inverse Document Frequency)
      - Weights words by importance
      - Rare words get higher weights
      - Better than BoW
      - Example: "I love ML" → [0.1, 0.8, 0.9, 0, 0, ...]
   
   c) WORD EMBEDDINGS
      - Dense vectors representing word meaning
      - Captures semantic relationships
      - Lower dimensional
      - Example: "love" → [0.2, -0.1, 0.5, ...] (300 dimensions)
   
   d) CONTEXTUAL EMBEDDINGS (Transformers)
      - Same word, different embeddings based on context
      - State-of-the-art
      - Example: "bank" (river) vs "bank" (financial)

3. KEY CONCEPTS:

   - SPARSE vs DENSE: BoW/TF-IDF are sparse, embeddings are dense
   - DIMENSIONALITY: BoW can be huge, embeddings are fixed size
   - SEMANTICS: Embeddings capture meaning, BoW doesn't
   - CONTEXT: Only contextual embeddings capture context
"""

print(theory)

# ============================================================================
# PRACTICE: Sample Text Data
# ============================================================================
print("\n" + "=" * 70)
print("PRACTICE: Working with Text Representations")
print("=" * 70)

# Sample documents
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with multiple layers",
    "Natural language processing helps computers understand text",
    "Large language models are trained on vast amounts of data",
    "Machine learning algorithms learn from data patterns",
    "Neural networks are inspired by the human brain",
    "Text processing involves tokenization and normalization",
    "Language models can generate human-like text"
]

print(f"\nSample Documents ({len(documents)}):")
for i, doc in enumerate(documents[:3], 1):
    print(f"{i}. {doc}")

# ============================================================================
# METHOD 1: Bag of Words (BoW)
# ============================================================================
print("\n" + "=" * 70)
print("Method 1: Bag of Words (BoW)")
print("=" * 70)

# Create BoW representation
vectorizer_bow = CountVectorizer(max_features=20, stop_words='english')
bow_matrix = vectorizer_bow.fit_transform(documents)

print(f"\nBoW Matrix Shape: {bow_matrix.shape}")
print(f"Vocabulary size: {len(vectorizer_bow.vocabulary_)}")
print(f"\nVocabulary (first 10):")
vocab = list(vectorizer_bow.vocabulary_.keys())[:10]
print(vocab)

# Convert to dense for visualization
bow_dense = bow_matrix.toarray()
print(f"\nBoW for first document:")
print(f"Document: {documents[0]}")
print(f"Vector: {bow_dense[0]}")
print(f"Non-zero elements: {np.count_nonzero(bow_dense[0])}")

# Visualize BoW
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.heatmap(bow_dense[:5, :10], annot=True, fmt='d', cmap='Blues',
            xticklabels=vocab[:10], yticklabels=[f"Doc {i+1}" for i in range(5)])
plt.title('Bag of Words (First 5 docs, 10 features)', fontweight='bold')
plt.ylabel('Documents')
plt.xlabel('Words')

# Word frequency
word_counts = bow_matrix.sum(axis=0).A1
top_words_idx = word_counts.argsort()[-10:][::-1]
top_words = [list(vectorizer_bow.vocabulary_.keys())[list(vectorizer_bow.vocabulary_.values()).index(idx)] 
             for idx in top_words_idx]

plt.subplot(1, 2, 2)
plt.barh(range(len(top_words)), word_counts[top_words_idx])
plt.yticks(range(len(top_words)), top_words)
plt.xlabel('Frequency')
plt.title('Top 10 Words (BoW)', fontweight='bold')
plt.tight_layout()
plt.savefig('bow_representation.png', dpi=150, bbox_inches='tight')
print("\nSaved: bow_representation.png")
plt.close()

# ============================================================================
# METHOD 2: TF-IDF
# ============================================================================
print("\n" + "=" * 70)
print("Method 2: TF-IDF (Term Frequency-Inverse Document Frequency)")
print("=" * 70)

# Create TF-IDF representation
vectorizer_tfidf = TfidfVectorizer(max_features=20, stop_words='english')
tfidf_matrix = vectorizer_tfidf.fit_transform(documents)

print(f"\nTF-IDF Matrix Shape: {tfidf_matrix.shape}")
print(f"\nTF-IDF for first document:")
print(f"Document: {documents[0]}")
tfidf_dense = tfidf_matrix.toarray()
print(f"Vector (first 10): {tfidf_dense[0, :10]}")
print(f"Max value: {tfidf_dense[0].max():.4f} (word: {vectorizer_tfidf.get_feature_names_out()[tfidf_dense[0].argmax()]})")

# Compare BoW vs TF-IDF
print("\nComparison: BoW vs TF-IDF")
print(f"BoW - 'learning' in doc 1: {bow_dense[0, list(vectorizer_bow.vocabulary_.values()).index(list(vectorizer_bow.vocabulary_.keys()).index('learning'))] if 'learning' in vectorizer_bow.vocabulary_ else 0}")
print(f"TF-IDF - 'learning' in doc 1: {tfidf_dense[0, list(vectorizer_tfidf.vocabulary_.values()).index(list(vectorizer_tfidf.vocabulary_.keys()).index('learning'))] if 'learning' in vectorizer_tfidf.vocabulary_ else 0:.4f}")

# Visualize TF-IDF
plt.figure(figsize=(12, 6))
sns.heatmap(tfidf_dense[:5, :10], annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=vectorizer_tfidf.get_feature_names_out()[:10],
            yticklabels=[f"Doc {i+1}" for i in range(5)])
plt.title('TF-IDF Representation (First 5 docs, 10 features)', fontweight='bold')
plt.ylabel('Documents')
plt.xlabel('Words')
plt.tight_layout()
plt.savefig('tfidf_representation.png', dpi=150, bbox_inches='tight')
print("\nSaved: tfidf_representation.png")
plt.close()

# ============================================================================
# METHOD 3: Word Embeddings (Word2Vec)
# ============================================================================
print("\n" + "=" * 70)
print("Method 3: Word Embeddings (Word2Vec)")
print("=" * 70)

# Prepare text for Word2Vec (tokenized sentences)
tokenized_docs = [doc.lower().split() for doc in documents]

print("\nTokenized documents (first 2):")
for i, tokens in enumerate(tokenized_docs[:2], 1):
    print(f"{i}. {tokens}")

# Train Word2Vec model
print("\nTraining Word2Vec model...")
w2v_model = Word2Vec(
    sentences=tokenized_docs,
    vector_size=100,      # Embedding dimension
    window=5,             # Context window
    min_count=1,          # Minimum word count
    workers=4,
    sg=0                  # 0 = CBOW, 1 = Skip-gram
)

print(f"\nWord2Vec Model:")
print(f"Vocabulary size: {len(w2v_model.wv)}")
print(f"Vector dimension: {w2v_model.wv.vector_size}")

# Get word embeddings
word = "learning"
if word in w2v_model.wv:
    embedding = w2v_model.wv[word]
    print(f"\nEmbedding for '{word}':")
    print(f"Shape: {embedding.shape}")
    print(f"First 10 values: {embedding[:10]}")

# Find similar words
print(f"\nWords similar to '{word}':")
if word in w2v_model.wv:
    similar = w2v_model.wv.most_similar(word, topn=5)
    for similar_word, score in similar:
        print(f"  {similar_word}: {score:.4f}")

# Visualize embeddings (2D using PCA)
from sklearn.decomposition import PCA

words_to_plot = ['machine', 'learning', 'neural', 'networks', 'language', 'text', 'data', 'models']
word_vectors = [w2v_model.wv[word] for word in words_to_plot if word in w2v_model.wv]
words_found = [word for word in words_to_plot if word in w2v_model.wv]

if len(word_vectors) > 0:
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(word_vectors)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], s=100, alpha=0.6)
    for i, word in enumerate(words_found):
        plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), fontsize=10)
    plt.title('Word Embeddings Visualization (2D PCA)', fontweight='bold')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('word_embeddings_2d.png', dpi=150, bbox_inches='tight')
    print("\nSaved: word_embeddings_2d.png")
    plt.close()

# ============================================================================
# METHOD 4: Pre-trained Embeddings
# ============================================================================
print("\n" + "=" * 70)
print("Method 4: Using Pre-trained Embeddings")
print("=" * 70)

print("\nLoading pre-trained GloVe embeddings (small sample)...")
print("Note: For full pre-trained models, use:")
print("  from gensim.downloader import load")
print("  model = load('glove-wiki-gigaword-100')")

# For demonstration, we'll use our trained model
# In practice, use pre-trained models like:
# - GloVe
# - Word2Vec (Google News)
# - FastText

print("\nPre-trained embeddings advantages:")
print("- Trained on large corpora (billions of words)")
print("- Better semantic understanding")
print("- Ready to use, no training needed")
print("- Captures relationships: king - man + woman ≈ queen")

# ============================================================================
# COMPARISON: All Methods
# ============================================================================
print("\n" + "=" * 70)
print("Comparison: All Representation Methods")
print("=" * 70)

comparison = pd.DataFrame({
    'Method': ['Bag of Words', 'TF-IDF', 'Word2Vec', 'Contextual Embeddings'],
    'Dimensionality': ['High (vocab size)', 'High (vocab size)', 'Low (100-300)', 'Low (768-4096)'],
    'Sparsity': ['Very Sparse', 'Sparse', 'Dense', 'Dense'],
    'Semantics': ['No', 'No', 'Yes', 'Yes'],
    'Context': ['No', 'No', 'No', 'Yes'],
    'Training': ['None', 'None', 'Required', 'Pre-trained'],
    'Use Case': ['Simple tasks', 'Better than BoW', 'General NLP', 'State-of-the-art']
})

print("\n" + comparison.to_string(index=False))

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================
print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
1. BoW: Simple, fast, but loses word order and semantics
2. TF-IDF: Better than BoW, weights important words
3. Word Embeddings: Captures semantics, dense vectors
4. Contextual Embeddings: Best performance, captures context

CHOOSING REPRESENTATION:
- Simple classification: TF-IDF
- Semantic tasks: Word embeddings
- State-of-the-art: Contextual embeddings (BERT, etc.)
- Quick prototyping: Pre-trained embeddings

NEXT STEPS:
- Try sentence embeddings
- Explore contextual embeddings (BERT)
- Learn about document embeddings
- Move to advanced NLP tasks
"""

print(takeaways)

print("\n" + "=" * 70)
print("Next: Explore advanced NLP with transformers")
print("=" * 70)

