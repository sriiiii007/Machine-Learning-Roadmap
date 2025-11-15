"""
Classification Basics - Your First Classification Model
Understanding classification vs regression in ML.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("CLASSIFICATION BASICS - Theory & Practice")
print("=" * 70)

# ============================================================================
# THEORY: Classification vs Regression
# ============================================================================
print("\n" + "=" * 70)
print("THEORY: Understanding Classification")
print("=" * 70)

theory = """
MACHINE LEARNING TYPES

1. SUPERVISED LEARNING:
   - Uses labeled data (we know the answers)
   - Learns from examples
   - Can predict on new data

   a) REGRESSION
      - Predicts continuous values (numbers)
      - Examples: Price, Temperature, Age
      - Output: Real numbers
   
   b) CLASSIFICATION
      - Predicts categories/classes (labels)
      - Examples: Spam/Not Spam, Cat/Dog, Disease/No Disease
      - Output: Categories

2. CLASSIFICATION TYPES:

   a) BINARY CLASSIFICATION
      - Two classes (Yes/No, 0/1, True/False)
      - Examples: Spam detection, Disease diagnosis
   
   b) MULTI-CLASS CLASSIFICATION
      - More than two classes
      - Examples: Image classification (Cat/Dog/Bird)
      - Handwritten digit recognition (0-9)

3. KEY CONCEPTS:

   - Features (X): Input variables
   - Target/Label (y): What we want to predict
   - Training: Learning from data
   - Prediction: Making predictions on new data
   - Evaluation: Measuring how good the model is

4. EVALUATION METRICS:

   - Accuracy: % of correct predictions
   - Precision: Of predicted positives, how many are actually positive
   - Recall: Of actual positives, how many did we catch
   - F1-Score: Balance of precision and recall
   - Confusion Matrix: Shows all predictions vs actual
"""

print(theory)

# ============================================================================
# PRACTICE: Binary Classification Example
# ============================================================================
print("\n" + "=" * 70)
print("PRACTICE: Email Spam Classification")
print("=" * 70)

# Create sample dataset
np.random.seed(42)
n_samples = 1000

# Features: email characteristics
# Feature 1: Number of exclamation marks (spam emails have more)
# Feature 2: Number of capital letters (spam emails have more)
# Feature 3: Email length (spam emails are often longer)

# Generate data
spam_emails = {
    'exclamation_marks': np.random.poisson(5, n_samples),  # Spam has more
    'capital_letters': np.random.poisson(20, n_samples),     # Spam has more
    'email_length': np.random.randint(200, 2000, n_samples)  # Spam is longer
}

ham_emails = {
    'exclamation_marks': np.random.poisson(1, n_samples),   # Normal has fewer
    'capital_letters': np.random.poisson(5, n_samples),      # Normal has fewer
    'email_length': np.random.randint(50, 500, n_samples)    # Normal is shorter
}

# Create DataFrame
spam_df = pd.DataFrame(spam_emails)
spam_df['label'] = 1  # 1 = Spam

ham_df = pd.DataFrame(ham_emails)
ham_df['label'] = 0  # 0 = Not Spam (Ham)

# Combine
data = pd.concat([spam_df, ham_df], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

print(f"\nDataset shape: {data.shape}")
print(f"\nLabel distribution:")
print(data['label'].value_counts())
print(f"\nFirst 5 rows:")
print(data.head())

# ============================================================================
# DATA EXPLORATION
# ============================================================================
print("\n" + "=" * 70)
print("Data Exploration")
print("=" * 70)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Feature 1: Exclamation marks
axes[0].hist(data[data['label']==0]['exclamation_marks'], alpha=0.7, label='Ham', bins=20)
axes[0].hist(data[data['label']==1]['exclamation_marks'], alpha=0.7, label='Spam', bins=20)
axes[0].set_xlabel('Exclamation Marks')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Exclamation Marks Distribution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Feature 2: Capital letters
axes[1].hist(data[data['label']==0]['capital_letters'], alpha=0.7, label='Ham', bins=20)
axes[1].hist(data[data['label']==1]['capital_letters'], alpha=0.7, label='Spam', bins=20)
axes[1].set_xlabel('Capital Letters')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Capital Letters Distribution')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Feature 3: Email length
axes[2].hist(data[data['label']==0]['email_length'], alpha=0.7, label='Ham', bins=20)
axes[2].hist(data[data['label']==1]['email_length'], alpha=0.7, label='Spam', bins=20)
axes[2].set_xlabel('Email Length')
axes[2].set_ylabel('Frequency')
axes[2].set_title('Email Length Distribution')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('classification_exploration.png', dpi=150, bbox_inches='tight')
print("\nSaved exploration plot: classification_exploration.png")
plt.close()

# ============================================================================
# PREPARE DATA
# ============================================================================
print("\n" + "=" * 70)
print("Preparing Data for ML")
print("=" * 70)

# Features and target
X = data[['exclamation_marks', 'capital_letters', 'email_length']]
y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Features: {X_train.shape[1]}")

# ============================================================================
# MODEL 1: Logistic Regression
# ============================================================================
print("\n" + "=" * 70)
print("Model 1: Logistic Regression")
print("=" * 70)

# Create and train model
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test)
y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]  # Probability of spam

# Evaluate
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"\nAccuracy: {accuracy_lr:.4f} ({accuracy_lr*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['Ham', 'Spam']))

# Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
print("\nConfusion Matrix:")
print("                Predicted")
print("              Ham    Spam")
print(f"Actual Ham   {cm_lr[0,0]:4d}   {cm_lr[0,1]:4d}")
print(f"       Spam  {cm_lr[1,0]:4d}   {cm_lr[1,1]:4d}")

# ============================================================================
# MODEL 2: Decision Tree
# ============================================================================
print("\n" + "=" * 70)
print("Model 2: Decision Tree")
print("=" * 70)

# Create and train model
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model.fit(X_train, y_train)

# Predictions
y_pred_dt = dt_model.predict(X_test)

# Evaluate
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"\nAccuracy: {accuracy_dt:.4f} ({accuracy_dt*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt, target_names=['Ham', 'Spam']))

# ============================================================================
# VISUALIZING RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("Visualizing Results")
print("=" * 70)

# Confusion Matrix Heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Logistic Regression
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
axes[0].set_title('Logistic Regression - Confusion Matrix', fontweight='bold')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# Decision Tree
cm_dt = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
axes[1].set_title('Decision Tree - Confusion Matrix', fontweight='bold')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('classification_results.png', dpi=150, bbox_inches='tight')
print("\nSaved results plot: classification_results.png")
plt.close()

# ============================================================================
# UNDERSTANDING PREDICTIONS
# ============================================================================
print("\n" + "=" * 70)
print("Understanding Predictions")
print("=" * 70)

# Show some predictions with probabilities
print("\nSample Predictions (Logistic Regression):")
sample_indices = np.random.choice(len(X_test), 10, replace=False)
sample_data = X_test.iloc[sample_indices].copy()
sample_data['Actual'] = y_test.iloc[sample_indices].values
sample_data['Predicted'] = y_pred_lr[sample_indices]
sample_data['Probability_Spam'] = y_pred_proba_lr[sample_indices]

print(sample_data[['exclamation_marks', 'capital_letters', 'email_length', 
                   'Actual', 'Predicted', 'Probability_Spam']])

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================
print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
1. Classification predicts categories (not numbers)
2. Binary classification has 2 classes
3. Accuracy measures % of correct predictions
4. Confusion matrix shows all prediction types
5. Different algorithms can give different results
6. Always evaluate on test set (not training set)

CLASSIFICATION METRICS:
- Accuracy: Overall correctness
- Precision: Of predicted positives, how many are correct
- Recall: Of actual positives, how many did we find
- F1-Score: Balance of precision and recall

COMMON CLASSIFICATION ALGORITHMS:
- Logistic Regression: Linear, fast, interpretable
- Decision Trees: Non-linear, interpretable
- Random Forest: Ensemble of trees, very accurate
- SVM: Good for complex boundaries
- Neural Networks: Very flexible, can be complex

NEXT STEPS:
- Try multi-class classification
- Experiment with different algorithms
- Learn about evaluation metrics in detail
- Move to core algorithms section
"""

print(takeaways)

print("\n" + "=" * 70)
print("Next: Explore core ML algorithms in 03_core_algorithms/")
print("=" * 70)

