"""
Multi-Language Sentiment Analysis Platform
Enterprise-level NLP project for sentiment analysis across multiple languages.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("MULTI-LANGUAGE SENTIMENT ANALYSIS PLATFORM")
print("=" * 70)

# ============================================================================
# DATA PREPARATION
# ============================================================================
print("\n" + "=" * 70)
print("Step 1: Data Preparation")
print("=" * 70)

def create_sample_data():
    """Create sample multilingual sentiment data"""
    np.random.seed(42)
    
    # English reviews
    english_positive = [
        "This product is amazing! I love it so much.",
        "Excellent quality, highly recommend to everyone.",
        "Best purchase I've made this year, worth every penny.",
        "Outstanding service and great customer support.",
        "Perfect for my needs, exactly what I was looking for.",
    ] * 200
    
    english_negative = [
        "Terrible product, complete waste of money.",
        "Poor quality, broke after just one week.",
        "Very disappointed, not worth the price at all.",
        "Awful customer service, would not recommend.",
        "Cheaply made, fell apart immediately.",
    ] * 200
    
    # Spanish reviews (translated examples)
    spanish_positive = [
        "Este producto es increíble! Me encanta mucho.",
        "Excelente calidad, altamente recomendado.",
        "La mejor compra que he hecho este año.",
        "Servicio excepcional y gran atención al cliente.",
        "Perfecto para mis necesidades.",
    ] * 200
    
    spanish_negative = [
        "Producto terrible, desperdicio total de dinero.",
        "Mala calidad, se rompió después de una semana.",
        "Muy decepcionado, no vale la pena.",
        "Pésimo servicio al cliente, no recomendaría.",
        "Hecho barato, se deshizo de inmediato.",
    ] * 200
    
    # Combine
    texts = english_positive + english_negative + spanish_positive + spanish_negative
    labels = [1] * 1000 + [0] * 1000
    languages = ['en'] * 1000 + ['es'] * 1000
    
    # Shuffle
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    languages = [languages[i] for i in indices]
    
    return pd.DataFrame({
        'text': texts,
        'label': labels,
        'language': languages
    })

# Load data
print("\nCreating sample dataset...")
df = create_sample_data()
print(f"Dataset shape: {df.shape}")
print(f"\nLanguage distribution:")
print(df['language'].value_counts())
print(f"\nSentiment distribution:")
print(df['label'].value_counts())

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================
print("\n" + "=" * 70)
print("Step 2: Text Preprocessing")
print("=" * 70)

def preprocess_text(text):
    """Preprocess text for sentiment analysis"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters (keep letters, numbers, spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove extra spaces
    text = ' '.join(text.split())
    return text

df['processed_text'] = df['text'].apply(preprocess_text)
print(f"\nSample preprocessing:")
print(f"Original: {df['text'].iloc[0]}")
print(f"Processed: {df['processed_text'].iloc[0]}")

# ============================================================================
# TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "=" * 70)
print("Step 3: Train-Test Split")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'], df['label'], 
    test_size=0.2, random_state=42, stratify=df['label']
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
print("\n" + "=" * 70)
print("Step 4: Feature Extraction (TF-IDF)")
print("=" * 70)

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',  # For English; in production, use language-specific
    ngram_range=(1, 2),     # Unigrams and bigrams
    min_df=2,
    max_df=0.95
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"TF-IDF features: {X_train_tfidf.shape[1]}")
print(f"Training matrix shape: {X_train_tfidf.shape}")
print(f"Test matrix shape: {X_test_tfidf.shape}")

# ============================================================================
# MODEL TRAINING
# ============================================================================
print("\n" + "=" * 70)
print("Step 5: Model Training")
print("=" * 70)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Naive Bayes': MultinomialNB()
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_tfidf, y_train)
    
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred
    }
    
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# ============================================================================
# PIPELINE APPROACH (Best Practice)
# ============================================================================
print("\n" + "=" * 70)
print("Step 6: Pipeline Approach")
print("=" * 70)

sentiment_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

sentiment_pipeline.fit(X_train, y_train)
y_pred_pipeline = sentiment_pipeline.predict(X_test)
accuracy_pipeline = accuracy_score(y_test, y_pred_pipeline)

print(f"Pipeline Accuracy: {accuracy_pipeline:.4f} ({accuracy_pipeline*100:.2f}%)")

# ============================================================================
# EVALUATION
# ============================================================================
print("\n" + "=" * 70)
print("Step 7: Detailed Evaluation")
print("=" * 70)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_pipeline, target_names=['Negative', 'Positive']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_pipeline)
print("\nConfusion Matrix:")
print("                Predicted")
print("              Negative  Positive")
print(f"Actual Negative   {cm[0,0]:4d}      {cm[0,1]:4d}")
print(f"       Positive   {cm[1,0]:4d}      {cm[1,1]:4d}")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "=" * 70)
print("Step 8: Visualization")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
axes[0].set_title('Confusion Matrix', fontweight='bold')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# Model Comparison
model_names = list(results.keys()) + ['Pipeline']
accuracies = [results[m]['accuracy'] for m in results.keys()] + [accuracy_pipeline]
colors = ['skyblue', 'lightcoral', 'lightgreen']
axes[1].bar(model_names, accuracies, color=colors, edgecolor='black', alpha=0.7)
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Model Comparison', fontweight='bold')
axes[1].set_ylim([0, 1])
for i, acc in enumerate(accuracies):
    axes[1].text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('sentiment_analysis_results.png', dpi=150, bbox_inches='tight')
print("\nSaved: sentiment_analysis_results.png")
plt.close()

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
print("\n" + "=" * 70)
print("Step 9: Prediction Function")
print("=" * 70)

def predict_sentiment(text, pipeline):
    """Predict sentiment for a text"""
    processed = preprocess_text(text)
    prediction = pipeline.predict([processed])[0]
    probability = pipeline.predict_proba([processed])[0]
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = probability[prediction] * 100
    
    return sentiment, confidence, probability

# Test predictions
test_texts = [
    "I love this product! It's amazing!",
    "This is terrible, worst purchase ever.",
    "It's okay, nothing special.",
    "Excellent quality and fast delivery!"
]

print("\nSample Predictions:")
for text in test_texts:
    sentiment, confidence, prob = predict_sentiment(text, sentiment_pipeline)
    print(f"\nText: {text}")
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Probabilities: Negative={prob[0]:.3f}, Positive={prob[1]:.3f}")

# ============================================================================
# MODEL SAVING
# ============================================================================
print("\n" + "=" * 70)
print("Step 10: Model Persistence")
print("=" * 70)

joblib.dump(sentiment_pipeline, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Saved model artifacts:")
print("  - sentiment_model.pkl")
print("  - vectorizer.pkl")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
✅ Sentiment Analysis Model Trained!
✅ Accuracy: {accuracy_pipeline:.4f} ({accuracy_pipeline*100:.2f}%)
✅ Model saved and ready for deployment

Next Steps:
1. Add more languages (use language detection)
2. Fine-tune with domain-specific data
3. Deploy as API (FastAPI)
4. Create real-time monitoring dashboard
5. Integrate with customer feedback systems
6. Add aspect-based sentiment analysis
""")

print("=" * 70)

