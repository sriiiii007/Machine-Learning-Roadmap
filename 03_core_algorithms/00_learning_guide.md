# Core ML Algorithms Learning Guide 🎓

Complete guide to mastering essential Machine Learning algorithms.

---

## 📚 What You'll Learn

This section covers the most important ML algorithms:

1. **Linear Regression** - Predicting continuous values
2. **Logistic Regression** - Binary classification
3. **Decision Trees** - Interpretable non-linear models
4. **Random Forest** - Ensemble method
5. **K-Nearest Neighbors** - Instance-based learning
6. **Support Vector Machines** - Maximum margin classifiers
7. **K-Means Clustering** - Unsupervised clustering
8. **PCA** - Dimensionality reduction
9. **Naive Bayes** - Probabilistic classifier
10. **Ridge/Lasso** - Regularized regression

---

## 🗺️ Learning Path

### Week 1: Linear Models

**Day 1-2: Linear Regression**
- [ ] Study `01_linear_regression.py`
- [ ] Understand simple vs multiple regression
- [ ] Learn evaluation metrics (MSE, R²)
- [ ] Practice with different datasets

**Day 3-4: Logistic Regression**
- [ ] Study `02_logistic_regression.py`
- [ ] Understand binary classification
- [ ] Learn about probabilities
- [ ] Practice with classification problems

**Day 5-7: Regularized Regression**
- [ ] Study `12_ridge_lasso_regression.py`
- [ ] Understand overfitting
- [ ] Learn Ridge vs Lasso
- [ ] Practice regularization

### Week 2: Tree-Based Models

**Day 1-3: Decision Trees**
- [ ] Study `03_decision_trees.py`
- [ ] Understand splitting criteria
- [ ] Learn about overfitting
- [ ] Practice interpretation

**Day 4-7: Random Forest**
- [ ] Study `04_random_forest.py`
- [ ] Understand ensemble methods
- [ ] Learn feature importance
- [ ] Compare with single trees

### Week 3: Other Algorithms

**Day 1-2: KNN**
- [ ] Study `05_knn.py`
- [ ] Understand distance metrics
- [ ] Learn about choosing k
- [ ] Practice with different k values

**Day 3-4: SVM**
- [ ] Study `06_svm.py`
- [ ] Understand maximum margin
- [ ] Learn about kernels
- [ ] Practice with different kernels

**Day 5-6: Naive Bayes**
- [ ] Study `11_naive_bayes.py`
- [ ] Understand Bayes' theorem
- [ ] Learn independence assumption
- [ ] Practice with text data

**Day 7: Comparison**
- [ ] Study `00_algorithm_comparison.py`
- [ ] Compare all algorithms
- [ ] Understand when to use which

### Week 4: Unsupervised Learning

**Day 1-3: K-Means**
- [ ] Study `07_kmeans_clustering.py`
- [ ] Understand clustering
- [ ] Learn about choosing k
- [ ] Practice with different datasets

**Day 4-6: PCA**
- [ ] Study `08_pca.py`
- [ ] Understand dimensionality reduction
- [ ] Learn about explained variance
- [ ] Practice visualization

**Day 7: Review**
- [ ] Review all algorithms
- [ ] Complete exercises
- [ ] Build projects

---

## 📁 Files in This Section

### Algorithm Files

1. **`01_linear_regression.py`**
   - Simple linear regression
   - Multiple linear regression
   - Polynomial regression
   - Evaluation metrics

2. **`02_logistic_regression.py`**
   - Binary classification
   - Probability predictions
   - Decision boundaries
   - Evaluation metrics

3. **`03_decision_trees.py`**
   - Tree construction
   - Splitting criteria
   - Overfitting prevention
   - Visualization

4. **`04_random_forest.py`**
   - Ensemble method
   - Bagging
   - Feature importance
   - Hyperparameter tuning

5. **`05_knn.py`**
   - Distance metrics
   - Choosing k
   - Classification and regression
   - Pros and cons

6. **`06_svm.py`**
   - Maximum margin
   - Kernels
   - Classification and regression
   - Hyperparameter tuning

7. **`07_kmeans_clustering.py`**
   - Clustering algorithm
   - Choosing k
   - Evaluation
   - Visualization

8. **`08_pca.py`**
   - Dimensionality reduction
   - Explained variance
   - Visualization
   - Applications

9. **`09_model_evaluation.py`**
   - Evaluation metrics
   - Cross-validation
   - Overfitting detection
   - Best practices

10. **`11_naive_bayes.py`**
    - Bayes' theorem
    - Different types
    - Text classification
    - Applications

11. **`12_ridge_lasso_regression.py`**
    - Regularization
    - Ridge vs Lasso
    - Hyperparameter tuning
    - Feature selection

### Comparison and Theory

- **`00_algorithm_comparison.py`**: Compare all algorithms
- **`THEORY.md`**: Complete theoretical guide

---

## 🎯 Learning Objectives

By the end of this section, you should be able to:

### Understand Algorithms
- ✅ How each algorithm works
- ✅ When to use which algorithm
- ✅ Pros and cons of each
- ✅ Mathematical foundations

### Implement Algorithms
- ✅ Use scikit-learn implementations
- ✅ Tune hyperparameters
- ✅ Evaluate performance
- ✅ Compare algorithms

### Apply Algorithms
- ✅ Choose right algorithm for problem
- ✅ Preprocess data appropriately
- ✅ Interpret results
- ✅ Debug issues

---

## 💡 How to Use These Files

### For Each Algorithm

1. **Read Theory First**
   - Understand the concept
   - Learn the mathematics
   - Know when to use it

2. **Run the Code**
   ```bash
   python 01_linear_regression.py
   ```
   - See it in action
   - Understand the output
   - Experiment with parameters

3. **Practice**
   - Try with different data
   - Modify parameters
   - Compare results

4. **Compare**
   - Use `00_algorithm_comparison.py`
   - See which works best
   - Understand tradeoffs

### Study Order

**Beginner Path**:
1. Linear Regression
2. Logistic Regression
3. Decision Trees
4. Random Forest
5. KNN
6. Others as needed

**Advanced Path**:
1. Review all algorithms
2. Deep dive into favorites
3. Understand mathematical details
4. Implement from scratch (optional)

---

## 🛠️ Setup

### Required Packages

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Verify Installation

```python
import sklearn
print(f"Scikit-learn version: {sklearn.__version__}")
```

---

## 📊 Practice Projects

### Beginner Projects

1. **House Price Prediction**
   - Use Linear Regression
   - Try Ridge/Lasso
   - Compare results

2. **Spam Detection**
   - Use Logistic Regression
   - Try Naive Bayes
   - Compare accuracy

3. **Iris Classification**
   - Try multiple algorithms
   - Compare performance
   - Visualize results

### Intermediate Projects

1. **Customer Segmentation**
   - Use K-Means
   - Choose optimal k
   - Interpret clusters

2. **Feature Selection**
   - Use Lasso
   - Compare with Ridge
   - Analyze coefficients

3. **Algorithm Comparison**
   - Test all algorithms
   - Create comparison report
   - Choose best model

---

## 🎓 Assessment

### Self-Check

After completing this section:

1. **Understanding**
   - [ ] Can I explain how each algorithm works?
   - [ ] Do I know when to use which?
   - [ ] Can I interpret results?

2. **Implementation**
   - [ ] Can I implement each algorithm?
   - [ ] Can I tune hyperparameters?
   - [ ] Can I evaluate performance?

3. **Application**
   - [ ] Can I choose the right algorithm?
   - [ ] Can I preprocess data correctly?
   - [ ] Can I debug issues?

---

## 🚀 Next Steps

After mastering core algorithms:

1. ✅ Move to Deep Learning
2. ✅ Learn advanced techniques
3. ✅ Build real-world projects
4. ✅ Explore specialized domains

---

## 📚 Additional Resources

### Books
- "Hands-On Machine Learning" by Aurélien Géron
- "Introduction to Statistical Learning" by James et al.

### Online Courses
- Scikit-learn documentation
- Kaggle Learn courses
- Coursera ML courses

### Practice
- Kaggle competitions
- UCI ML Repository
- Scikit-learn examples

---

## 💪 Tips for Success

1. **Understand First**: Don't just copy code
2. **Experiment**: Try different parameters
3. **Compare**: Always compare multiple algorithms
4. **Practice**: Work with real datasets
5. **Document**: Keep notes on what works

---

## ❓ Common Questions

**Q: Which algorithm should I use?**
A: Start with simple ones (Linear/Logistic Regression), then try tree-based (Random Forest), compare and choose.

**Q: Do I need to understand the math?**
A: Basic understanding helps, but you can use algorithms without deep math knowledge.

**Q: How do I choose hyperparameters?**
A: Use cross-validation, try different values, use GridSearchCV.

**Q: Which algorithm is best?**
A: Depends on your problem. Random Forest often works well, but always compare.

---

**Remember**: Master these algorithms - they form the foundation of all ML work! 🎉

---

**Ready to start?** Open `01_linear_regression.py` and begin! 🚀

