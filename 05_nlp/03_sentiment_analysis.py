"""
Sentiment Analysis - Complete Example
Building a sentiment analysis system from scratch.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("SENTIMENT ANALYSIS - Complete Example")
print("=" * 70)

# ============================================================================
# THEORY: Sentiment Analysis
# ============================================================================
print("\n" + "=" * 70)
print("THEORY: Understanding Sentiment Analysis")
print("=" * 70)

theory = """
SENTIMENT ANALYSIS

1. WHAT IS IT?
   - Classify text as positive, negative, or neutral
   - Understand opinions and emotions
   - Applications: Reviews, social media, customer feedback

2. APPROACHES:

   a) RULE-BASED
      - Use predefined rules
      - Simple but limited
   
   b) MACHINE LEARNING
      - Train on labeled data
      - More accurate
      - Can learn patterns
   
   c) DEEP LEARNING
      - Neural networks
      - Best performance
      - Requires more data

3. TYPES:
   - Binary: Positive/Negative
   - Multi-class: Positive/Neutral/Negative
   - Fine-grained: 1-5 stars, emotions

4. CHALLENGES:
   - Sarcasm and irony
   - Context dependency
   - Domain-specific language
   - Mixed sentiments
"""

print(theory)

# ============================================================================
# PRACTICE: Create Sample Dataset
# ============================================================================
print("\n" + "=" * 70)
print("PRACTICE: Building Sentiment Analysis System")
print("=" * 70)

# Create sample review dataset
np.random.seed(42)

positive_reviews = [
    "This product is amazing! I love it so much.",
    "Excellent quality, highly recommend to everyone.",
    "Best purchase I've made this year, worth every penny.",
    "Outstanding service and great customer support.",
    "Perfect for my needs, exactly what I was looking for.",
    "Great value for money, very satisfied with purchase.",
    "Top quality product, exceeded my expectations.",
    "Wonderful experience, will definitely buy again.",
    "Fantastic product, works exactly as described.",
    "Highly satisfied, great quality and fast delivery."
]

negative_reviews = [
    "Terrible product, complete waste of money.",
    "Poor quality, broke after just one week.",
    "Very disappointed, not worth the price at all.",
    "Awful customer service, would not recommend.",
    "Cheaply made, fell apart immediately.",
    "Worst purchase ever, regret buying this.",
    "Low quality materials, very unsatisfied.",
    "Does not work as advertised, false claims.",
    "Poor design, many issues and problems.",
    "Terrible experience, will never buy again."
]

# Create dataset
reviews = positive_reviews + negative_reviews
labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)

# Add some variation
reviews = reviews * 10  # Repeat to have more data
labels = labels * 10

# Shuffle
indices = np.random.permutation(len(reviews))
reviews = [reviews[i] for i in indices]
labels = [labels[i] for i in indices]

print(f"\nDataset created:")
print(f"Total reviews: {len(reviews)}")
print(f"Positive: {sum(labels)}")
print(f"Negative: {len(labels) - sum(labels)}")

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 70)
print("Data Preprocessing")
print("=" * 70)

def simple_preprocess(text):
    """Simple text preprocessing"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters (keep spaces and letters)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = ' '.join(text.split())
    return text

# Preprocess reviews
processed_reviews = [simple_preprocess(review) for review in reviews]

print(f"\nSample preprocessing:")
print(f"Original: {reviews[0]}")
print(f"Processed: {processed_reviews[0]}")

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
print("\n" + "=" * 70)
print("Feature Extraction (TF-IDF)")
print("=" * 70)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    processed_reviews, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"\nTF-IDF features:")
print(f"Train shape: {X_train_tfidf.shape}")
print(f"Test shape: {X_test_tfidf.shape}")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

# ============================================================================
# MODEL 1: Logistic Regression
# ============================================================================
print("\n" + "=" * 70)
print("Model 1: Logistic Regression")
print("=" * 70)

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

y_pred_lr = lr_model.predict(X_test_tfidf)
y_pred_proba_lr = lr_model.predict_proba(X_test_tfidf)[:, 1]

accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"\nAccuracy: {accuracy_lr:.4f} ({accuracy_lr*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['Negative', 'Positive']))

# ============================================================================
# MODEL 2: Naive Bayes
# ============================================================================
print("\n" + "=" * 70)
print("Model 2: Naive Bayes")
print("=" * 70)

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

y_pred_nb = nb_model.predict(X_test_tfidf)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"\nAccuracy: {accuracy_nb:.4f} ({accuracy_nb*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_nb, target_names=['Negative', 'Positive']))

# ============================================================================
# PIPELINE APPROACH
# ============================================================================
print("\n" + "=" * 70)
print("Pipeline Approach (Best Practice)")
print("=" * 70)

# Create pipeline
sentiment_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Train pipeline
sentiment_pipeline.fit(X_train, y_train)

# Predict
y_pred_pipeline = sentiment_pipeline.predict(X_test)
accuracy_pipeline = accuracy_score(y_test, y_pred_pipeline)

print(f"\nPipeline Accuracy: {accuracy_pipeline:.4f} ({accuracy_pipeline*100:.2f}%)")

# Test on new reviews
new_reviews = [
    "This is the best product ever!",
    "I hate this, terrible quality.",
    "It's okay, nothing special."
]

print("\nPredictions on new reviews:")
for review in new_reviews:
    pred = sentiment_pipeline.predict([review])[0]
    proba = sentiment_pipeline.predict_proba([review])[0]
    sentiment = "Positive" if pred == 1 else "Negative"
    confidence = proba[pred] * 100
    print(f"\nReview: {review}")
    print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f}%)")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "=" * 70)
print("Visualizing Results")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
axes[0].set_title('Confusion Matrix', fontweight='bold')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# Model Comparison
models = ['Logistic\nRegression', 'Naive\nBayes', 'Pipeline']
accuracies = [accuracy_lr, accuracy_nb, accuracy_pipeline]
colors = ['skyblue', 'lightcoral', 'lightgreen']

axes[1].bar(models, accuracies, color=colors, edgecolor='black', alpha=0.7)
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Model Comparison', fontweight='bold')
axes[1].set_ylim([0, 1])
axes[1].grid(True, alpha=0.3, axis='y')

for i, acc in enumerate(accuracies):
    axes[1].text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('sentiment_analysis_results.png', dpi=150, bbox_inches='tight')
print("\nSaved: sentiment_analysis_results.png")
plt.close()

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================
print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
1. Sentiment analysis classifies text emotions
2. TF-IDF is good for traditional ML approaches
3. Logistic Regression works well for sentiment
4. Pipelines make deployment easier
5. Preprocessing is crucial

IMPROVEMENTS:
- Use more training data
- Try word embeddings (Word2Vec, GloVe)
- Use deep learning (LSTM, BERT)
- Handle sarcasm and context
- Domain-specific training

NEXT STEPS:
- Try with real datasets (IMDB, Amazon reviews)
- Experiment with different features
- Try deep learning approaches
- Build production system
"""

print(takeaways)

print("\n" + "=" * 70)
print("Next: Explore advanced NLP with transformers and BERT")
print("=" * 70)

