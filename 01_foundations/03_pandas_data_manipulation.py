"""
Pandas for Data Manipulation - Complete Guide
Essential pandas operations for machine learning.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("PANDAS DATA MANIPULATION - Theory & Practice")
print("=" * 70)

# ============================================================================
# THEORY: What is Pandas?
# ============================================================================
print("\n" + "=" * 70)
print("THEORY: Understanding Pandas")
print("=" * 70)

theory = """
PANDAS (Python Data Analysis Library)

1. WHAT IS PANDAS?
   - Built on NumPy
   - Provides DataFrame (like Excel spreadsheet)
   - Essential for data manipulation and analysis
   - Used in 90% of ML data preprocessing

2. KEY DATA STRUCTURES:

   a) SERIES
      - 1-dimensional labeled array
      - Like a column in Excel
      - Has index and values
   
   b) DATAFRAME
      - 2-dimensional labeled data structure
      - Like Excel spreadsheet
      - Rows and columns
      - Most commonly used

3. WHY PANDAS?
   - Easy data loading (CSV, JSON, Excel, SQL)
   - Data cleaning and preprocessing
   - Data manipulation (filter, group, merge)
   - Missing data handling
   - Time series operations

4. COMMON OPERATIONS:
   - Reading/writing data
   - Selecting and filtering
   - Grouping and aggregation
   - Merging and joining
   - Handling missing values
   - Data transformation
"""

print(theory)

# ============================================================================
# PRACTICE: Creating DataFrames
# ============================================================================
print("\n" + "=" * 70)
print("PRACTICE: Creating DataFrames")
print("=" * 70)

# 1. From dictionary
print("\n1. Creating DataFrame from dictionary:")
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 28, 32],
    'City': ['NYC', 'LA', 'Chicago', 'NYC', 'LA'],
    'Salary': [50000, 60000, 70000, 55000, 65000]
}
df = pd.DataFrame(data)
print(df)
print(f"\nShape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# 2. From CSV (simulated)
print("\n2. Creating sample data for ML:")
np.random.seed(42)
ml_data = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'feature3': np.random.randn(100),
    'target': np.random.randint(0, 2, 100)
})
print(f"ML Dataset shape: {ml_data.shape}")
print(f"\nFirst 5 rows:\n{ml_data.head()}")

# ============================================================================
# DATA EXPLORATION
# ============================================================================
print("\n" + "=" * 70)
print("Data Exploration (Essential for ML)")
print("=" * 70)

print("\n1. Basic Information:")
print(f"Shape: {df.shape}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nInfo:\n{df.info()}")

print("\n2. Statistical Summary:")
print(df.describe())

print("\n3. First/Last rows:")
print(f"First 3 rows:\n{df.head(3)}")
print(f"\nLast 3 rows:\n{df.tail(3)}")

print("\n4. Value counts:")
print(f"City counts:\n{df['City'].value_counts()}")

# ============================================================================
# SELECTING DATA
# ============================================================================
print("\n" + "=" * 70)
print("Selecting and Filtering Data")
print("=" * 70)

print("\n1. Selecting columns:")
print(f"Single column:\n{df['Name']}")
print(f"\nMultiple columns:\n{df[['Name', 'Age']]}")

print("\n2. Selecting rows:")
print(f"First 3 rows:\n{df.iloc[0:3]}")
print(f"\nRows by label:\n{df.loc[0:2]}")

print("\n3. Filtering:")
print(f"Age > 28:\n{df[df['Age'] > 28]}")
print(f"\nCity == 'NYC':\n{df[df['City'] == 'NYC']}")
print(f"\nMultiple conditions:\n{df[(df['Age'] > 28) & (df['City'] == 'NYC')]}")

# ============================================================================
# DATA MANIPULATION
# ============================================================================
print("\n" + "=" * 70)
print("Data Manipulation")
print("=" * 70)

# Adding columns
print("\n1. Adding columns:")
df['Age_Group'] = df['Age'].apply(lambda x: 'Young' if x < 30 else 'Old')
print(df)

# Modifying columns
print("\n2. Modifying columns:")
df['Salary_K'] = df['Salary'] / 1000
print(df[['Name', 'Salary', 'Salary_K']])

# Dropping columns
print("\n3. Dropping columns:")
df_dropped = df.drop('Salary_K', axis=1)
print(f"Columns after drop: {df_dropped.columns.tolist()}")

# Sorting
print("\n4. Sorting:")
print(f"Sorted by Age:\n{df.sort_values('Age')}")
print(f"\nSorted by Salary (descending):\n{df.sort_values('Salary', ascending=False)}")

# ============================================================================
# HANDLING MISSING DATA (Critical for ML)
# ============================================================================
print("\n" + "=" * 70)
print("Handling Missing Data (Critical for ML)")
print("=" * 70)

# Create data with missing values
df_missing = df.copy()
df_missing.loc[0, 'Age'] = np.nan
df_missing.loc[2, 'Salary'] = np.nan
df_missing.loc[4, 'City'] = None

print("\n1. Detecting missing values:")
print(f"Missing values:\n{df_missing.isnull().sum()}")
print(f"\nData with missing:\n{df_missing}")

print("\n2. Handling missing values:")
print("Option 1: Drop rows with missing values")
df_dropped = df_missing.dropna()
print(f"After dropna:\n{df_dropped}")

print("\nOption 2: Fill missing values")
df_filled = df_missing.fillna({'Age': df_missing['Age'].mean(), 
                                'Salary': df_missing['Salary'].median(),
                                'City': 'Unknown'})
print(f"After fillna:\n{df_filled}")

# ============================================================================
# GROUPING AND AGGREGATION
# ============================================================================
print("\n" + "=" * 70)
print("Grouping and Aggregation")
print("=" * 70)

print("\n1. Group by City:")
grouped = df.groupby('City')
print(f"Groups: {grouped.groups}")

print("\n2. Aggregations:")
print(f"Mean salary by city:\n{grouped['Salary'].mean()}")
print(f"\nCount by city:\n{grouped.size()}")
print(f"\nMultiple aggregations:\n{grouped['Salary'].agg(['mean', 'min', 'max', 'count'])}")

# ============================================================================
# MERGING DATA
# ============================================================================
print("\n" + "=" * 70)
print("Merging DataFrames")
print("=" * 70)

# Create second DataFrame
df2 = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Department': ['IT', 'HR', 'IT'],
    'Manager': ['John', 'Jane', 'John']
})

print("\nDataFrame 1:")
print(df[['Name', 'City', 'Salary']])
print("\nDataFrame 2:")
print(df2)

print("\nMerged DataFrame:")
merged = pd.merge(df, df2, on='Name', how='inner')
print(merged)

# ============================================================================
# PRACTICAL ML EXAMPLE: Data Preprocessing Pipeline
# ============================================================================
print("\n" + "=" * 70)
print("Practical ML Example: Complete Data Preprocessing")
print("=" * 70)

# Create realistic ML dataset
np.random.seed(42)
ml_df = pd.DataFrame({
    'age': np.random.randint(18, 65, 1000),
    'income': np.random.normal(50000, 15000, 1000),
    'education_years': np.random.randint(12, 20, 1000),
    'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], 1000),
    'purchased': np.random.choice([0, 1], 1000, p=[0.6, 0.4])
})

# Add some missing values
ml_df.loc[np.random.choice(ml_df.index, 50), 'income'] = np.nan
ml_df.loc[np.random.choice(ml_df.index, 30), 'age'] = np.nan

print("\n1. Original Data:")
print(f"Shape: {ml_df.shape}")
print(f"Missing values:\n{ml_df.isnull().sum()}")
print(f"\nFirst 5 rows:\n{ml_df.head()}")

# Data cleaning
print("\n2. Data Cleaning:")
ml_df_clean = ml_df.copy()
ml_df_clean['income'].fillna(ml_df_clean['income'].median(), inplace=True)
ml_df_clean['age'].fillna(ml_df_clean['age'].median(), inplace=True)

print(f"Missing values after cleaning:\n{ml_df_clean.isnull().sum()}")

# Feature engineering
print("\n3. Feature Engineering:")
ml_df_clean['income_per_age'] = ml_df_clean['income'] / ml_df_clean['age']
ml_df_clean['high_education'] = (ml_df_clean['education_years'] >= 16).astype(int)

print(f"New features created: income_per_age, high_education")
print(f"\nUpdated columns: {ml_df_clean.columns.tolist()}")

# Encoding categorical variables
print("\n4. Encoding Categorical Variables:")
ml_df_encoded = pd.get_dummies(ml_df_clean, columns=['city'], prefix='city')
print(f"After encoding:\n{ml_df_encoded[['city_NYC', 'city_LA', 'city_Chicago', 'city_Houston']].head()}")

# Normalization
print("\n5. Normalization:")
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numeric_cols = ['age', 'income', 'education_years', 'income_per_age']
ml_df_encoded[numeric_cols] = scaler.fit_transform(ml_df_encoded[numeric_cols])

print(f"Normalized numeric columns:\n{ml_df_encoded[numeric_cols].head()}")

# Final dataset
print("\n6. Final Preprocessed Dataset:")
print(f"Shape: {ml_df_encoded.shape}")
print(f"Columns: {ml_df_encoded.columns.tolist()}")
print(f"\nReady for ML model!")

# ============================================================================
# READING AND WRITING DATA
# ============================================================================
print("\n" + "=" * 70)
print("Reading and Writing Data")
print("=" * 70)

print("""
Common file formats:

1. CSV:
   df = pd.read_csv('file.csv')
   df.to_csv('output.csv', index=False)

2. Excel:
   df = pd.read_excel('file.xlsx')
   df.to_excel('output.xlsx', index=False)

3. JSON:
   df = pd.read_json('file.json')
   df.to_json('output.json')

4. SQL:
   df = pd.read_sql('SELECT * FROM table', connection)
   df.to_sql('table', connection)

5. Parquet (efficient):
   df = pd.read_parquet('file.parquet')
   df.to_parquet('output.parquet')
""")

# Save example
df.to_csv('sample_data.csv', index=False)
print("\nSaved sample data to 'sample_data.csv'")

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================
print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
1. Pandas is essential for data preprocessing in ML
2. DataFrame is the main data structure (like Excel)
3. Always explore data first (head, describe, info)
4. Handle missing values before modeling
5. Encode categorical variables (one-hot encoding)
6. Normalize/scale numeric features
7. Feature engineering can improve model performance

COMMON PANDAS OPERATIONS FOR ML:
- pd.read_csv(): Load data
- df.head(), df.describe(): Explore data
- df.isnull(), df.fillna(): Handle missing values
- pd.get_dummies(): Encode categorical variables
- df.groupby(): Group and aggregate
- df.merge(): Combine datasets
- df.drop(), df.dropna(): Remove data
- df.sort_values(): Sort data

NEXT STEPS:
- Practice with real datasets
- Learn data visualization (Matplotlib/Seaborn)
- Move to ML model building
"""

print(takeaways)

print("\n" + "=" * 70)
print("Next: Learn data visualization with Matplotlib")
print("=" * 70)

