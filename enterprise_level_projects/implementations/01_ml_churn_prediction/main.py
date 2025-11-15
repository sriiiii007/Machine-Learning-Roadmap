"""
Customer Churn Prediction System
Enterprise-level ML project for predicting customer churn.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("CUSTOMER CHURN PREDICTION SYSTEM")
print("=" * 70)

# ============================================================================
# DATA LOADING AND EXPLORATION
# ============================================================================
print("\n" + "=" * 70)
print("Step 1: Data Loading and Exploration")
print("=" * 70)

def load_data():
    """Load or generate sample churn data"""
    # In production, load from database or file
    # For demo, we'll generate synthetic data
    np.random.seed(42)
    n_samples = 10000
    
    data = pd.DataFrame({
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 80, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'tenure': np.random.randint(0, 72, n_samples),  # months
        'monthly_charges': np.random.normal(50, 20, n_samples),
        'total_charges': np.random.normal(2000, 1000, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                         n_samples, p=[0.5, 0.3, 0.2]),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 
                                          'Bank transfer', 'Credit card'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], 
                                            n_samples, p=[0.4, 0.4, 0.2]),
        'phone_service': np.random.choice(['Yes', 'No'], n_samples),
        'multiple_lines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    })
    
    # Create target variable (churn) based on features
    # Higher churn for: month-to-month, high charges, low tenure
    churn_prob = (
        0.3 * (data['contract_type'] == 'Month-to-month').astype(int) +
        0.2 * (data['monthly_charges'] > data['monthly_charges'].quantile(0.75)).astype(int) +
        0.2 * (data['tenure'] < 12).astype(int) +
        0.1 * (data['online_security'] == 'No').astype(int) +
        np.random.rand(n_samples) * 0.2
    )
    data['churn'] = (churn_prob > 0.5).astype(int)
    
    return data

# Load data
print("\nLoading data...")
df = load_data()
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nChurn distribution:")
print(df['churn'].value_counts())
print(f"Churn rate: {df['churn'].mean()*100:.2f}%")

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 70)
print("Step 2: Data Preprocessing")
print("=" * 70)

def preprocess_data(df):
    """Preprocess data for ML"""
    df_processed = df.copy()
    
    # Drop customer_id (not a feature)
    df_processed = df_processed.drop('customer_id', axis=1)
    
    # Encode categorical variables
    categorical_cols = ['gender', 'contract_type', 'payment_method', 
                       'internet_service', 'phone_service', 'multiple_lines',
                       'online_security', 'tech_support']
    
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        le_dict[col] = le
    
    # Separate features and target
    X = df_processed.drop('churn', axis=1)
    y = df_processed['churn']
    
    return X, y, le_dict

X, y, label_encoders = preprocess_data(df)
print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFeature columns: {list(X.columns)}")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 70)
print("Step 3: Feature Engineering")
print("=" * 70)

# Add new features
X['charges_per_tenure'] = X['monthly_charges'] / (X['tenure'] + 1)
X['high_value_customer'] = ((X['monthly_charges'] > X['monthly_charges'].quantile(0.75)) & 
                            (X['tenure'] > X['tenure'].median())).astype(int)

print(f"Added features: charges_per_tenure, high_value_customer")
print(f"New feature shape: {X.shape}")

# ============================================================================
# TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "=" * 70)
print("Step 4: Train-Test Split")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Train churn rate: {y_train.mean()*100:.2f}%")
print(f"Test churn rate: {y_test.mean()*100:.2f}%")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# MODEL TRAINING
# ============================================================================
print("\n" + "=" * 70)
print("Step 5: Model Training")
print("=" * 70)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

print("\nTraining models...")
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Use scaled data for Logistic Regression
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")

# ============================================================================
# MODEL SELECTION
# ============================================================================
print("\n" + "=" * 70)
print("Step 6: Model Selection")
print("=" * 70)

best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_model_name]['model']

print(f"\nBest Model: {best_model_name}")
print(f"F1-Score: {results[best_model_name]['f1']:.4f}")
print(f"ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")

# ============================================================================
# HYPERPARAMETER TUNING (Optional)
# ============================================================================
print("\n" + "=" * 70)
print("Step 7: Hyperparameter Tuning (Optional)")
print("=" * 70)

print("\nTuning Random Forest...")
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_grid = GridSearchCV(rf_base, rf_param_grid, cv=5, scoring='f1', n_jobs=-1)
rf_grid.fit(X_train, y_train)

print(f"Best parameters: {rf_grid.best_params_}")
print(f"Best CV score: {rf_grid.best_score_:.4f}")

# Update best model
best_model = rf_grid.best_estimator_
y_pred_tuned = best_model.predict(X_test)
y_pred_proba_tuned = best_model.predict_proba(X_test)[:, 1]

print(f"\nTuned Model Performance:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_tuned):.4f}")
print(f"  F1-Score: {f1_score(y_test, y_pred_tuned):.4f}")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 70)
print("Step 8: Feature Importance Analysis")
print("=" * 70)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Visualize
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Feature Importance', fontweight='bold')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
print("\nSaved: feature_importance.png")
plt.close()

# ============================================================================
# EVALUATION VISUALIZATION
# ============================================================================
print("\n" + "=" * 70)
print("Step 9: Evaluation Visualization")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_tuned)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
axes[0, 0].set_title('Confusion Matrix', fontweight='bold')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].set_xlabel('Predicted')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_tuned)
axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc_score(y_test, y_pred_proba_tuned):.3f})')
axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Model Comparison
model_names = list(results.keys())
f1_scores = [results[m]['f1'] for m in model_names]
axes[1, 0].bar(model_names, f1_scores, color=['skyblue', 'lightcoral', 'lightgreen'])
axes[1, 0].set_ylabel('F1-Score')
axes[1, 0].set_title('Model Comparison', fontweight='bold')
axes[1, 0].set_ylim([0, 1])
for i, v in enumerate(f1_scores):
    axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

# Probability Distribution
axes[1, 1].hist(y_pred_proba_tuned[y_test == 0], bins=20, alpha=0.7, 
                label='No Churn', color='green', density=True)
axes[1, 1].hist(y_pred_proba_tuned[y_test == 1], bins=20, alpha=0.7, 
                label='Churn', color='red', density=True)
axes[1, 1].set_xlabel('Churn Probability')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Probability Distribution', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=150, bbox_inches='tight')
print("\nSaved: model_evaluation.png")
plt.close()

# ============================================================================
# MODEL SAVING
# ============================================================================
print("\n" + "=" * 70)
print("Step 10: Model Persistence")
print("=" * 70)

# Save model and preprocessing objects
joblib.dump(best_model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(feature_importance, 'feature_importance.pkl')

print("Saved model artifacts:")
print("  - churn_model.pkl")
print("  - scaler.pkl")
print("  - label_encoders.pkl")
print("  - feature_importance.pkl")

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
print("\n" + "=" * 70)
print("Step 11: Prediction Function")
print("=" * 70)

def predict_churn(customer_data, model, scaler, label_encoders):
    """Predict churn for a customer"""
    # Preprocess
    df = pd.DataFrame([customer_data])
    
    # Encode categorical
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))
    
    # Feature engineering
    if 'monthly_charges' in df.columns and 'tenure' in df.columns:
        df['charges_per_tenure'] = df['monthly_charges'] / (df['tenure'] + 1)
    
    # Scale
    X = df.drop('churn', axis=1, errors='ignore')
    X_scaled = scaler.transform(X)
    
    # Predict
    probability = model.predict_proba(X_scaled)[0, 1]
    prediction = 1 if probability > 0.5 else 0
    
    return prediction, probability

# Example prediction
example_customer = {
    'age': 45,
    'gender': 'M',
    'tenure': 12,
    'monthly_charges': 70,
    'total_charges': 840,
    'contract_type': 'Month-to-month',
    'payment_method': 'Electronic check',
    'internet_service': 'Fiber optic',
    'phone_service': 'Yes',
    'multiple_lines': 'No',
    'online_security': 'No',
    'tech_support': 'No'
}

pred, prob = predict_churn(example_customer, best_model, scaler, label_encoders)
print(f"\nExample Prediction:")
print(f"Customer: {example_customer}")
print(f"Predicted Churn: {'Yes' if pred == 1 else 'No'}")
print(f"Churn Probability: {prob:.4f} ({prob*100:.2f}%)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
✅ Model trained successfully!
✅ Best Model: {best_model_name}
✅ F1-Score: {results[best_model_name]['f1']:.4f}
✅ ROC-AUC: {results[best_model_name]['roc_auc']:.4f}

Next Steps:
1. Deploy model as API (FastAPI)
2. Create dashboard for monitoring
3. Set up retraining pipeline
4. Implement alert system for high-risk customers
5. A/B test retention campaigns
""")

print("=" * 70)

