"""
Machine Learning Core Concepts
Understanding the fundamental concepts of ML.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("MACHINE LEARNING CORE CONCEPTS - Theory & Practice")
print("=" * 70)

# ============================================================================
# THEORY: Core ML Concepts
# ============================================================================
print("\n" + "=" * 70)
print("THEORY: Understanding ML Concepts")
print("=" * 70)

theory = """
MACHINE LEARNING CORE CONCEPTS

1. SUPERVISED vs UNSUPERVISED LEARNING:

   SUPERVISED LEARNING:
   - Uses labeled data (we know the answers)
   - Learns from examples
   - Can predict on new data
   - Types: Regression, Classification
   
   UNSUPERVISED LEARNING:
   - Uses unlabeled data (no answers)
   - Finds patterns in data
   - Types: Clustering, Dimensionality Reduction

2. TRAIN/TEST SPLIT:

   Why Split?
   - Train on training set
   - Evaluate on test set (unseen data)
   - Prevents overfitting assessment
   
   Common Split:
   - 80% training, 20% testing
   - 70% training, 15% validation, 15% testing
   - Use random_state for reproducibility

3. OVERFITTING vs UNDERFITTING:

   OVERFITTING:
   - Model learns training data too well
   - Poor generalization to new data
   - High training accuracy, low test accuracy
   - Solution: Regularization, more data, simpler model
   
   UNDERFITTING:
   - Model too simple
   - Can't learn patterns
   - Low training and test accuracy
   - Solution: More complex model, more features

4. BIAS-VARIANCE TRADEOFF:

   BIAS: Error from oversimplifying
   VARIANCE: Error from sensitivity to fluctuations
   
   Goal: Balance bias and variance
   - High bias: Underfitting
   - High variance: Overfitting

5. CROSS-VALIDATION:

   Why?
   - Better estimate of model performance
   - Uses all data for training and testing
   - Reduces variance in performance estimate
   
   K-Fold Cross-Validation:
   - Split data into k folds
   - Train on k-1 folds, test on 1 fold
   - Repeat k times
   - Average the results
"""

print(theory)

# ============================================================================
# PRACTICE: Train/Test Split
# ============================================================================
print("\n" + "=" * 70)
print("PRACTICE: Understanding Train/Test Split")
print("=" * 70)

# Create sample data
np.random.seed(42)
X = np.random.randn(1000, 3)
y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(1000) * 0.1

print(f"\nOriginal data shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nAfter split:")
print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"\nModel Performance:")
print(f"Training R²: {train_score:.4f}")
print(f"Test R²: {test_score:.4f}")

# ============================================================================
# PRACTICE: Overfitting Demonstration
# ============================================================================
print("\n" + "=" * 70)
print("PRACTICE: Demonstrating Overfitting")
print("=" * 70)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Create data with some noise
np.random.seed(42)
X_simple = np.linspace(0, 10, 50).reshape(-1, 1)
y_simple = 2 * X_simple.flatten() + 3 + np.random.randn(50) * 2

# Split
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
    X_simple, y_simple, test_size=0.3, random_state=42
)

# Models with different complexity
degrees = [1, 3, 10, 20]
train_scores = []
test_scores = []

print("\nTraining models with different complexity...")
for degree in degrees:
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train_simple)
    X_test_poly = poly.transform(X_test_simple)
    
    # Train
    model = LinearRegression()
    model.fit(X_train_poly, y_train_simple)
    
    # Evaluate
    train_score = model.score(X_train_poly, y_train_simple)
    test_score = model.score(X_test_poly, y_test_simple)
    
    train_scores.append(train_score)
    test_scores.append(test_score)
    
    print(f"Degree {degree:2d}: Train R² = {train_score:.4f}, Test R² = {test_score:.4f}")

# Visualize
plt.figure(figsize=(12, 5))

# Plot 1: Scores
plt.subplot(1, 2, 1)
plt.plot(degrees, train_scores, 'o-', label='Training Score', linewidth=2, markersize=8)
plt.plot(degrees, test_scores, 's-', label='Test Score', linewidth=2, markersize=8)
plt.xlabel('Polynomial Degree (Model Complexity)', fontsize=12)
plt.ylabel('R² Score', fontsize=12)
plt.title('Overfitting: Training vs Test Performance', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Plot 2: Model fits
plt.subplot(1, 2, 2)
X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
plt.scatter(X_train_simple, y_train_simple, alpha=0.6, label='Training Data', s=50)
plt.scatter(X_test_simple, y_test_simple, alpha=0.6, label='Test Data', s=50, marker='s')

# Plot polynomial fits
for degree in [1, 10]:
    poly = PolynomialFeatures(degree=degree)
    X_plot_poly = poly.fit_transform(X_plot)
    model = LinearRegression()
    X_train_poly = poly.fit_transform(X_train_simple)
    model.fit(X_train_poly, y_train_simple)
    y_plot = model.predict(X_plot_poly)
    plt.plot(X_plot, y_plot, label=f'Degree {degree}', linewidth=2, alpha=0.7)

plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('Model Fits: Simple vs Complex', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('overfitting_demo.png', dpi=150, bbox_inches='tight')
print("\nSaved: overfitting_demo.png")
plt.close()

print("\nObservation:")
print("- Low degree (1): Underfitting (both scores low)")
print("- Medium degree (3): Good balance")
print("- High degree (10+): Overfitting (train high, test low)")

# ============================================================================
# PRACTICE: Cross-Validation
# ============================================================================
print("\n" + "=" * 70)
print("PRACTICE: Cross-Validation")
print("=" * 70)

# Create data
np.random.seed(42)
X_cv = np.random.randn(200, 5)
y_cv = 2 * X_cv[:, 0] + 3 * X_cv[:, 1] + np.random.randn(200) * 0.5

# Single train/test split
X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(
    X_cv, y_cv, test_size=0.2, random_state=42
)

model_single = LinearRegression()
model_single.fit(X_train_cv, y_train_cv)
single_score = model_single.score(X_test_cv, y_test_cv)

print(f"\nSingle Train/Test Split:")
print(f"Test R²: {single_score:.4f}")

# Cross-validation
cv_scores = cross_val_score(LinearRegression(), X_cv, y_cv, cv=5, scoring='r2')

print(f"\n5-Fold Cross-Validation:")
print(f"Scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.4f}")
print(f"Std: {cv_scores.std():.4f}")

print("\nWhy Cross-Validation?")
print("- More reliable performance estimate")
print("- Uses all data for both training and testing")
print("- Reduces variance in performance estimate")

# ============================================================================
# PRACTICE: Feature Scaling
# ============================================================================
print("\n" + "=" * 70)
print("PRACTICE: Feature Scaling (Normalization)")
print("=" * 70)

# Create data with different scales
np.random.seed(42)
X_scaled = np.column_stack([
    np.random.randn(100) * 100,      # Large scale
    np.random.randn(100) * 0.1,       # Small scale
    np.random.randn(100) * 10         # Medium scale
])
y_scaled = 2 * X_scaled[:, 0] + 3 * X_scaled[:, 1] + X_scaled[:, 2] + np.random.randn(100)

print("\nOriginal Features:")
print(f"Feature 1 - Mean: {X_scaled[:, 0].mean():.2f}, Std: {X_scaled[:, 0].std():.2f}")
print(f"Feature 2 - Mean: {X_scaled[:, 1].mean():.2f}, Std: {X_scaled[:, 1].std():.2f}")
print(f"Feature 3 - Mean: {X_scaled[:, 2].mean():.2f}, Std: {X_scaled[:, 2].std():.2f}")

# Without scaling
model_no_scale = LinearRegression()
model_no_scale.fit(X_scaled, y_scaled)
score_no_scale = model_no_scale.score(X_scaled, y_scaled)

# With scaling
scaler = StandardScaler()
X_scaled_norm = scaler.fit_transform(X_scaled)
model_scaled = LinearRegression()
model_scaled.fit(X_scaled_norm, y_scaled)
score_scaled = model_scaled.score(X_scaled_norm, y_scaled)

print(f"\nAfter Scaling:")
print(f"Feature 1 - Mean: {X_scaled_norm[:, 0].mean():.2f}, Std: {X_scaled_norm[:, 0].std():.2f}")
print(f"Feature 2 - Mean: {X_scaled_norm[:, 1].mean():.2f}, Std: {X_scaled_norm[:, 1].std():.2f}")
print(f"Feature 3 - Mean: {X_scaled_norm[:, 2].mean():.2f}, Std: {X_scaled_norm[:, 2].std():.2f}")

print(f"\nModel Performance:")
print(f"Without scaling R²: {score_no_scale:.4f}")
print(f"With scaling R²: {score_scaled:.4f}")

print("\nNote: For this simple case, scaling may not matter much.")
print("But for algorithms like SVM, KNN, Neural Networks, scaling is critical!")

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================
print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
1. Always split data into train/test sets
2. Overfitting: Model too complex, learns noise
3. Underfitting: Model too simple, can't learn patterns
4. Cross-validation gives better performance estimates
5. Feature scaling important for some algorithms

BEST PRACTICES:
- Use train/test split (80/20 or 70/15/15)
- Use cross-validation for better estimates
- Scale features for algorithms that need it
- Monitor for overfitting (train vs test performance)
- Start simple, add complexity gradually

COMMON MISTAKES:
- Evaluating on training data (gives false confidence)
- Not splitting data before preprocessing
- Overfitting to training data
- Ignoring feature scaling
- Not using cross-validation

NEXT STEPS:
- Learn about different algorithms
- Practice with real datasets
- Understand evaluation metrics
- Move to core algorithms section
"""

print(takeaways)

print("\n" + "=" * 70)
print("Next: Explore core ML algorithms")
print("=" * 70)

