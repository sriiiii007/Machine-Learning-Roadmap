# Core ML Algorithms - Complete Theory Guide 📚

Comprehensive theoretical understanding of essential Machine Learning algorithms.

---

## Table of Contents

1. [Linear Regression](#linear-regression)
2. [Logistic Regression](#logistic-regression)
3. [Decision Trees](#decision-trees)
4. [Random Forest](#random-forest)
5. [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
6. [Support Vector Machines (SVM)](#support-vector-machines-svm)
7. [K-Means Clustering](#k-means-clustering)
8. [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
9. [Naive Bayes](#naive-bayes)
10. [Ridge and Lasso Regression](#ridge-and-lasso-regression)

---

## Linear Regression

### What is Linear Regression?

**Linear Regression** predicts a continuous target variable using a linear relationship with features.

### Mathematical Foundation

**Simple Linear Regression**:
```
y = β₀ + β₁x + ε
```

Where:
- `y`: Target variable
- `x`: Feature
- `β₀`: Intercept (bias)
- `β₁`: Slope (coefficient)
- `ε`: Error term

**Multiple Linear Regression**:
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

### How It Works

1. **Assumes linear relationship** between features and target
2. **Finds best line** that minimizes prediction error
3. **Uses least squares** to find optimal coefficients

### Cost Function

**Mean Squared Error (MSE)**:
```
MSE = (1/n) Σ(y_pred - y_true)²
```

**Goal**: Minimize MSE to find best coefficients

### Advantages

- Simple and interpretable
- Fast training and prediction
- No hyperparameters to tune
- Works well when relationship is linear

### Disadvantages

- Assumes linear relationship
- Sensitive to outliers
- Can't capture non-linear patterns
- Assumes features are independent

### When to Use

- Linear relationship exists
- Need interpretability
- Fast predictions needed
- Baseline model

---

## Logistic Regression

### What is Logistic Regression?

**Logistic Regression** predicts probabilities for binary classification using the logistic function.

### Mathematical Foundation

**Logistic Function (Sigmoid)**:
```
σ(z) = 1 / (1 + e^(-z))
```

Where `z = β₀ + β₁x₁ + ... + βₙxₙ`

**Output**: Probability between 0 and 1

### How It Works

1. **Calculates linear combination** of features
2. **Applies sigmoid function** to get probability
3. **Classifies** based on threshold (usually 0.5)

### Decision Boundary

- **Linear decision boundary**
- Separates classes with a line/plane
- Can be extended with polynomial features

### Advantages

- Probabilistic output
- Interpretable coefficients
- Fast and efficient
- No hyperparameters

### Disadvantages

- Assumes linear decision boundary
- Requires feature scaling
- Can't handle non-linear relationships well

### When to Use

- Binary classification
- Need probabilities
- Interpretability important
- Linear decision boundary sufficient

---

## Decision Trees

### What is a Decision Tree?

**Decision Trees** make predictions by asking a series of yes/no questions about features.

### How It Works

1. **Start at root** with all data
2. **Find best split** (feature + threshold)
3. **Split data** into branches
4. **Repeat** for each branch
5. **Stop** when stopping criteria met
6. **Predict** using leaf node

### Splitting Criteria

**Gini Impurity**:
```
Gini = 1 - Σ(pᵢ)²
```
- Measures impurity
- Lower is better
- Range: 0 (pure) to 0.5 (impure)

**Entropy**:
```
Entropy = -Σ(pᵢ * log₂(pᵢ))
```
- Measures information
- Lower is better
- Range: 0 (pure) to 1 (impure)

**Information Gain**:
```
IG = Entropy(parent) - Weighted Average Entropy(children)
```
- Measures improvement from split
- Higher is better

### Advantages

- Highly interpretable
- Handles non-linear relationships
- No feature scaling needed
- Handles mixed data types
- Feature importance available

### Disadvantages

- Prone to overfitting
- Unstable (small data changes → different tree)
- Biased toward features with more levels
- Can't capture additive relationships well

### When to Use

- Need interpretability
- Non-linear relationships
- Mixed data types
- Want feature importance

---

## Random Forest

### What is Random Forest?

**Random Forest** is an ensemble of decision trees that vote on the final prediction.

### How It Works

1. **Bootstrap sampling**: Create multiple datasets
2. **Train tree on each**: Different data, different features
3. **Vote on prediction**: Majority vote (classification) or average (regression)

### Key Concepts

**Bootstrap Aggregating (Bagging)**:
- Sample with replacement
- Each tree sees different data
- Reduces variance

**Feature Randomness**:
- Random subset of features at each split
- Prevents overfitting
- Increases diversity

### Advantages

- Reduces overfitting (vs single tree)
- Handles non-linear relationships
- Feature importance
- Works well out of the box
- Handles missing values

### Disadvantages

- Less interpretable than single tree
- Slower than single tree
- More memory usage
- Can overfit with noisy data

### When to Use

- Need better accuracy than single tree
- Non-linear relationships
- Want feature importance
- Can handle slower predictions

---

## K-Nearest Neighbors (KNN)

### What is KNN?

**KNN** classifies/predicts based on the k nearest neighbors in feature space.

### How It Works

1. **Store all training data** (lazy learning)
2. **Calculate distance** to new point
3. **Find k nearest neighbors**
4. **Predict**:
   - Classification: Majority vote
   - Regression: Average

### Distance Metrics

**Euclidean Distance**:
```
d = √(Σ(xᵢ - yᵢ)²)
```

**Manhattan Distance**:
```
d = Σ|xᵢ - yᵢ|
```

### Choosing K

- **Small k**: More sensitive to noise, lower bias, higher variance
- **Large k**: Smoother boundaries, higher bias, lower variance
- **Rule of thumb**: k = √n (where n is number of samples)

### Advantages

- Simple and intuitive
- No training phase
- Works for classification and regression
- Can learn complex patterns

### Disadvantages

- Slow prediction (must compute distances)
- Sensitive to irrelevant features
- Sensitive to scale
- Memory intensive (stores all data)

### When to Use

- Small to medium datasets
- Non-linear relationships
- Local patterns important
- Can handle slow predictions

---

## Support Vector Machines (SVM)

### What is SVM?

**SVM** finds the optimal hyperplane that separates classes with maximum margin.

### How It Works

1. **Find optimal hyperplane** with maximum margin
2. **Support vectors**: Points closest to hyperplane
3. **Margin**: Distance between hyperplane and nearest points
4. **Maximize margin** for best generalization

### Key Concepts

**Hard Margin**: No misclassifications allowed
**Soft Margin**: Allows some misclassifications (C parameter)

**Kernel Trick**: Maps data to higher dimensions
- Linear kernel
- Polynomial kernel
- RBF (Radial Basis Function) kernel

### Advantages

- Effective in high dimensions
- Memory efficient (uses support vectors only)
- Versatile (different kernels)
- Works well with clear margin

### Disadvantages

- Doesn't perform well with large datasets
- Sensitive to feature scaling
- Doesn't provide probabilities directly
- Black box model

### When to Use

- Clear margin of separation
- High-dimensional data
- Non-linear relationships (with kernels)
- Small to medium datasets

---

## K-Means Clustering

### What is K-Means?

**K-Means** groups data into k clusters based on similarity.

### How It Works

1. **Initialize k centroids** randomly
2. **Assign points** to nearest centroid
3. **Update centroids** to cluster means
4. **Repeat** until convergence

### Algorithm

```
1. Choose k centroids randomly
2. For each point:
   - Find nearest centroid
   - Assign to cluster
3. For each cluster:
   - Update centroid to mean
4. Repeat steps 2-3 until convergence
```

### Choosing K

**Elbow Method**:
- Plot within-cluster sum of squares (WCSS) vs k
- Look for "elbow" in plot

**Silhouette Score**:
- Measures how similar points are to their cluster
- Range: -1 to 1 (higher is better)

### Advantages

- Simple and fast
- Works well with spherical clusters
- Scales to large datasets
- Easy to implement

### Disadvantages

- Need to specify k
- Assumes spherical clusters
- Sensitive to initialization
- Sensitive to outliers

### When to Use

- Unsupervised learning
- Known number of clusters
- Spherical clusters
- Exploratory analysis

---

## Principal Component Analysis (PCA)

### What is PCA?

**PCA** reduces dimensionality by finding principal components (directions of maximum variance).

### How It Works

1. **Standardize data** (mean=0, std=1)
2. **Calculate covariance matrix**
3. **Find eigenvectors** (principal components)
4. **Project data** onto principal components
5. **Select top k components**

### Mathematical Foundation

**Covariance Matrix**:
```
C = (1/n) X^T X
```

**Eigenvalue Decomposition**:
```
C = P Λ P^T
```

Where:
- P: Principal components (eigenvectors)
- Λ: Eigenvalues (variance explained)

### Explained Variance

- Each component explains certain % of variance
- Cumulative variance shows information retained
- Choose number of components based on variance threshold

### Advantages

- Reduces dimensionality
- Removes correlation
- Reduces overfitting
- Speeds up training

### Disadvantages

- Loses interpretability
- Assumes linear relationships
- Sensitive to scaling
- May lose important information

### When to Use

- High-dimensional data
- Remove multicollinearity
- Visualize high-dimensional data
- Speed up algorithms

---

## Naive Bayes

### What is Naive Bayes?

**Naive Bayes** is a probabilistic classifier based on Bayes' theorem with "naive" independence assumption.

### Bayes' Theorem

```
P(A|B) = P(B|A) * P(A) / P(B)
```

**For Classification**:
```
P(Class|Features) = P(Features|Class) * P(Class) / P(Features)
```

### Naive Assumption

**Assumes features are independent**:
```
P(Features|Class) = P(x₁|Class) * P(x₂|Class) * ... * P(xₙ|Class)
```

### Types

**Gaussian Naive Bayes**: Continuous features (assumes normal distribution)
**Multinomial Naive Bayes**: Count data (text classification)
**Bernoulli Naive Bayes**: Binary features

### Advantages

- Fast training and prediction
- Works well with small datasets
- Handles multiple classes
- Probabilistic output
- Good baseline

### Disadvantages

- Strong independence assumption (often violated)
- Can be outperformed by other algorithms
- Requires feature independence

### When to Use

- Text classification
- Small datasets
- Need fast predictions
- Baseline model
- Features relatively independent

---

## Ridge and Lasso Regression

### What are Ridge and Lasso?

**Regularized linear regression** that prevents overfitting by adding penalty terms.

### Ridge Regression (L2 Regularization)

**Cost Function**:
```
MSE + α * Σ(βᵢ)²
```

**Effect**:
- Shrinks coefficients toward zero
- Doesn't eliminate features
- Prevents large coefficients

### Lasso Regression (L1 Regularization)

**Cost Function**:
```
MSE + α * Σ|βᵢ|
```

**Effect**:
- Can eliminate features (coefficients = 0)
- Feature selection
- Sparse models

### Comparison

| Aspect | Ridge | Lasso |
|--------|-------|-------|
| Penalty | L2 (squared) | L1 (absolute) |
| Feature Selection | No | Yes |
| Coefficients | Shrinks | Can eliminate |
| Use Case | Many features | Feature selection needed |

### Choosing α (Alpha)

- **Large α**: More regularization, simpler model
- **Small α**: Less regularization, more complex model
- **α = 0**: Regular linear regression

### Advantages

- Prevents overfitting
- Handles multicollinearity (Ridge)
- Feature selection (Lasso)
- Better generalization

### Disadvantages

- Need to tune α
- Lasso can eliminate important features
- Less interpretable than regular regression

### When to Use

- Many features
- Overfitting concerns
- Multicollinearity (Ridge)
- Feature selection needed (Lasso)

---

## Algorithm Selection Guide

### For Regression

- **Linear Regression**: Baseline, interpretable
- **Ridge/Lasso**: Many features, overfitting
- **Random Forest**: Non-linear, good performance
- **SVM**: High dimensions, clear patterns

### For Classification

- **Logistic Regression**: Baseline, interpretable
- **Decision Trees**: Interpretable, non-linear
- **Random Forest**: Best performance, feature importance
- **SVM**: High dimensions, clear margins
- **KNN**: Local patterns, simple

### For Clustering

- **K-Means**: Spherical clusters, fast
- **Hierarchical**: Unknown number of clusters
- **DBSCAN**: Arbitrary shapes, noise

---

## Best Practices

1. **Start Simple**: Begin with linear/logistic regression
2. **Compare Models**: Try multiple algorithms
3. **Tune Hyperparameters**: Optimize performance
4. **Cross-Validate**: Reliable performance estimates
5. **Feature Engineering**: Often more important than algorithm choice
6. **Ensemble Methods**: Combine models for better performance

---

## Common Mistakes

1. ❌ Not scaling features (SVM, KNN)
2. ❌ Using wrong algorithm for problem
3. ❌ Not tuning hyperparameters
4. ❌ Overfitting to training data
5. ❌ Ignoring feature engineering
6. ❌ Not comparing multiple algorithms

---

**Remember**: No single algorithm is best for all problems. Experiment, compare, and choose based on your specific needs!

