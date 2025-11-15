"""
Algorithm Comparison - Comparing Multiple ML Algorithms
Understanding when to use which algorithm.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, classification_report, 
                            mean_squared_error, r2_score, confusion_matrix)
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ML ALGORITHM COMPARISON - Theory & Practice")
print("=" * 70)

# ============================================================================
# THEORY: Algorithm Selection
# ============================================================================
print("\n" + "=" * 70)
print("THEORY: Choosing the Right Algorithm")
print("=" * 70)

theory = """
ALGORITHM SELECTION GUIDE

1. PROBLEM TYPE:
   - Regression: Predict continuous values
   - Classification: Predict categories
   - Clustering: Find groups (unsupervised)

2. DATA CHARACTERISTICS:
   - Size: Small vs Large
   - Features: Few vs Many
   - Linearity: Linear vs Non-linear
   - Quality: Clean vs Noisy

3. REQUIREMENTS:
   - Interpretability: Need to explain?
   - Speed: Fast predictions needed?
   - Accuracy: Maximum performance?
   - Scalability: Large datasets?

4. ALGORITHM COMPARISON:

   LINEAR MODELS (Fast, Interpretable):
   - Linear/Logistic Regression
   - Ridge/Lasso Regression
   - Good for: Linear relationships, baseline
   
   TREE-BASED (Non-linear, Interpretable):
   - Decision Trees
   - Random Forest
   - Good for: Non-linear, feature importance
   
   INSTANCE-BASED (Non-linear, Slow):
   - K-Nearest Neighbors
   - Good for: Local patterns, small datasets
   
   SUPPORT VECTORS (Non-linear, Complex):
   - SVM
   - Good for: High dimensions, clear margins
   
   PROBABILISTIC (Fast, Simple):
   - Naive Bayes
   - Good for: Text, small datasets, baseline
"""

print(theory)

# ============================================================================
# PRACTICE: Classification Comparison
# ============================================================================
print("\n" + "=" * 70)
print("PRACTICE: Comparing Classification Algorithms")
print("=" * 70)

# Create classification dataset
np.random.seed(42)
n_samples = 1000

# Generate non-linear classification data
X_class = np.random.randn(n_samples, 2)
y_class = ((X_class[:, 0]**2 + X_class[:, 1]**2) > 2).astype(int)

# Add some noise
y_class = y_class ^ (np.random.rand(n_samples) < 0.1).astype(int)

# Split data
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

# Scale features (important for some algorithms)
scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

# Initialize models
models_clf = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42, probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB()
}

# Train and evaluate
results_clf = {}
print("\nTraining and evaluating classification models...\n")

for name, model in models_clf.items():
    # Use scaled data for algorithms that need it
    if name in ['Logistic Regression', 'SVM', 'KNN']:
        X_train_use = X_train_clf_scaled
        X_test_use = X_test_clf_scaled
    else:
        X_train_use = X_train_clf
        X_test_use = X_test_clf
    
    # Train
    model.fit(X_train_use, y_train_clf)
    
    # Predict
    y_pred = model.predict(X_test_use)
    
    # Evaluate
    accuracy = accuracy_score(y_test_clf, y_pred)
    cv_scores = cross_val_score(model, X_train_use, y_train_clf, cv=5)
    
    results_clf[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'model': model
    }
    
    print(f"{name:20s}: Accuracy = {accuracy:.4f}, CV = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================================
# PRACTICE: Regression Comparison
# ============================================================================
print("\n" + "=" * 70)
print("PRACTICE: Comparing Regression Algorithms")
print("=" * 70)

# Create regression dataset with non-linear relationship
np.random.seed(42)
X_reg = np.random.randn(500, 3)
y_reg = 2 * X_reg[:, 0] + 3 * X_reg[:, 1]**2 - X_reg[:, 2] + np.random.randn(500) * 0.5

# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Scale features
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# Initialize models
models_reg = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=5),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVM': SVR(kernel='rbf'),
    'KNN': KNeighborsRegressor(n_neighbors=5)
}

# Train and evaluate
results_reg = {}
print("\nTraining and evaluating regression models...\n")

for name, model in models_reg.items():
    # Use scaled data for algorithms that need it
    if name in ['Ridge Regression', 'Lasso Regression', 'SVM', 'KNN']:
        X_train_use = X_train_reg_scaled
        X_test_use = X_test_reg_scaled
    else:
        X_train_use = X_train_reg
        X_test_use = X_test_reg
    
    # Train
    model.fit(X_train_use, y_train_reg)
    
    # Predict
    y_pred = model.predict(X_test_use)
    
    # Evaluate
    mse = mean_squared_error(y_test_reg, y_pred)
    r2 = r2_score(y_test_reg, y_pred)
    cv_scores = cross_val_score(model, X_train_use, y_train_reg, cv=5, scoring='r2')
    
    results_reg[name] = {
        'mse': mse,
        'r2': r2,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'model': model
    }
    
    print(f"{name:20s}: R² = {r2:.4f}, CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================================
# VISUALIZATION: Results Comparison
# ============================================================================
print("\n" + "=" * 70)
print("Visualizing Results")
print("=" * 70)

# Classification results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Classification Accuracy
clf_names = list(results_clf.keys())
clf_acc = [results_clf[name]['accuracy'] for name in clf_names]
clf_cv = [results_clf[name]['cv_mean'] for name in clf_names]

axes[0, 0].barh(clf_names, clf_acc, alpha=0.7, label='Test Accuracy')
axes[0, 0].barh(clf_names, clf_cv, alpha=0.7, label='CV Accuracy', left=clf_acc)
axes[0, 0].set_xlabel('Accuracy')
axes[0, 0].set_title('Classification: Test vs Cross-Validation Accuracy', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis='x')

# 2. Regression R²
reg_names = list(results_reg.keys())
reg_r2 = [results_reg[name]['r2'] for name in reg_names]
reg_cv = [results_reg[name]['cv_mean'] for name in reg_names]

axes[0, 1].barh(reg_names, reg_r2, alpha=0.7, label='Test R²')
axes[0, 1].barh(reg_names, reg_cv, alpha=0.7, label='CV R²', left=reg_r2)
axes[0, 1].set_xlabel('R² Score')
axes[0, 1].set_title('Regression: Test vs Cross-Validation R²', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='x')

# 3. Classification Confusion Matrix (Best model)
best_clf_name = max(results_clf, key=lambda x: results_clf[x]['accuracy'])
best_clf_model = results_clf[best_clf_name]['model']

if best_clf_name in ['Logistic Regression', 'SVM', 'KNN']:
    y_pred_best = best_clf_model.predict(X_test_clf_scaled)
else:
    y_pred_best = best_clf_model.predict(X_test_clf)

cm = confusion_matrix(y_test_clf, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
axes[1, 0].set_title(f'Confusion Matrix: {best_clf_name}', fontweight='bold')
axes[1, 0].set_ylabel('Actual')
axes[1, 0].set_xlabel('Predicted')

# 4. Regression Predictions (Best model)
best_reg_name = max(results_reg, key=lambda x: results_reg[x]['r2'])
best_reg_model = results_reg[best_reg_name]['model']

if best_reg_name in ['Ridge Regression', 'Lasso Regression', 'SVM', 'KNN']:
    y_pred_best_reg = best_reg_model.predict(X_test_reg_scaled)
else:
    y_pred_best_reg = best_reg_model.predict(X_test_reg)

axes[1, 1].scatter(y_test_reg, y_pred_best_reg, alpha=0.6)
axes[1, 1].plot([y_test_reg.min(), y_test_reg.max()], 
                [y_test_reg.min(), y_test_reg.max()], 'r--', linewidth=2)
axes[1, 1].set_xlabel('Actual')
axes[1, 1].set_ylabel('Predicted')
axes[1, 1].set_title(f'Predictions: {best_reg_name}', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('algorithm_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved: algorithm_comparison.png")
plt.close()

# ============================================================================
# KEY INSIGHTS
# ============================================================================
print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)

print(f"\nBest Classification Model: {best_clf_name}")
print(f"  Accuracy: {results_clf[best_clf_name]['accuracy']:.4f}")
print(f"  CV Score: {results_clf[best_clf_name]['cv_mean']:.4f} ± {results_clf[best_clf_name]['cv_std']:.4f}")

print(f"\nBest Regression Model: {best_reg_name}")
print(f"  R² Score: {results_reg[best_reg_name]['r2']:.4f}")
print(f"  CV Score: {results_reg[best_reg_name]['cv_mean']:.4f} ± {results_reg[best_reg_name]['cv_std']:.4f}")

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================
print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
1. No single algorithm is best for all problems
2. Always compare multiple algorithms
3. Use cross-validation for reliable estimates
4. Feature scaling matters for some algorithms
5. Tree-based models often perform well
6. Linear models are good baselines
7. Ensemble methods (Random Forest) often win

ALGORITHM SELECTION TIPS:
- Start with simple models (Linear/Logistic Regression)
- Try tree-based models (Decision Tree, Random Forest)
- Use cross-validation to compare
- Consider interpretability vs accuracy tradeoff
- Scale features for algorithms that need it

WHEN TO USE WHICH:
- Linear Models: Baseline, interpretable, fast
- Tree Models: Non-linear, interpretable, good performance
- SVM: High dimensions, clear margins
- KNN: Local patterns, small datasets
- Naive Bayes: Text, fast baseline
- Random Forest: Often best performance

NEXT STEPS:
- Tune hyperparameters for best models
- Try ensemble methods
- Experiment with feature engineering
- Move to advanced topics
"""

print(takeaways)

print("\n" + "=" * 70)
print("Next: Deep dive into individual algorithms")
print("=" * 70)

