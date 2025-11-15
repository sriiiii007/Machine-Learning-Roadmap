# Foundations - Complete Theory Guide 📚

Comprehensive theoretical understanding of Python, NumPy, Pandas, and Data Visualization for Machine Learning.

---

## Table of Contents

1. [Python Fundamentals](#python-fundamentals)
2. [NumPy Essentials](#numpy-essentials)
3. [Pandas for Data Science](#pandas-for-data-science)
4. [Data Visualization](#data-visualization)
5. [Data Preprocessing](#data-preprocessing)

---

## Python Fundamentals

### Why Python for ML?

- **Simple Syntax**: Easy to learn and read
- **Rich Ecosystem**: NumPy, Pandas, Scikit-learn, TensorFlow, PyTorch
- **Community**: Large community, lots of resources
- **Versatile**: Can do everything from data cleaning to deployment

### Key Python Concepts for ML

#### 1. Data Types

**Numeric Types**:
- `int`: Integers (1, 2, 3)
- `float`: Floating point (1.5, 3.14)
- `complex`: Complex numbers (rarely used in ML)

**Sequence Types**:
- `list`: Mutable, ordered collection `[1, 2, 3]`
- `tuple`: Immutable, ordered collection `(1, 2, 3)`
- `str`: String `"hello"`

**Mapping Type**:
- `dict`: Key-value pairs `{'key': 'value'}`

**Set Type**:
- `set`: Unordered, unique elements `{1, 2, 3}`

#### 2. List Comprehensions

**Syntax**: `[expression for item in iterable if condition]`

**Examples**:
```python
# Squares
squares = [x**2 for x in range(10)]

# Even numbers
evens = [x for x in range(20) if x % 2 == 0]

# Nested
matrix = [[i*j for j in range(3)] for i in range(3)]
```

**Why Important**: Faster and more Pythonic than loops

#### 3. Functions

**Definition**:
```python
def function_name(parameters):
    """Docstring"""
    # Code
    return value
```

**Key Concepts**:
- **Parameters**: Inputs to function
- **Return**: Output from function
- **Docstring**: Documentation
- **Scope**: Local vs global variables

#### 4. Lambda Functions

**Syntax**: `lambda arguments: expression`

**Example**:
```python
square = lambda x: x**2
sorted_list = sorted(data, key=lambda x: x[1])
```

**Use Cases**: Quick functions, sorting, filtering

---

## NumPy Essentials

### What is NumPy?

**NumPy** (Numerical Python) is the foundation of all scientific computing in Python.

### Key Concepts

#### 1. Arrays vs Lists

**Python Lists**:
- Can contain different types
- Slower for numerical operations
- More memory overhead

**NumPy Arrays**:
- Homogeneous (same type)
- Much faster (C implementation)
- Less memory
- Vectorized operations

#### 2. Array Creation

```python
# From list
arr = np.array([1, 2, 3])

# Special arrays
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
full = np.full((2, 2), 7)
identity = np.eye(3)

# Ranges
arange = np.arange(0, 10, 2)
linspace = np.linspace(0, 1, 5)
```

#### 3. Array Properties

- **Shape**: Dimensions `arr.shape`
- **Dtype**: Data type `arr.dtype`
- **Size**: Total elements `arr.size`
- **Ndims**: Number of dimensions `arr.ndim`

#### 4. Indexing and Slicing

**Basic Indexing**:
```python
arr[0]           # First element
arr[0, 0]         # First row, first column
arr[-1]           # Last element
```

**Slicing**:
```python
arr[0:3]          # First 3 elements
arr[:, 0]         # First column
arr[0:2, 1:3]     # Submatrix
```

**Boolean Indexing**:
```python
mask = arr > 5
arr[mask]         # Elements > 5
```

#### 5. Array Operations

**Mathematical**:
```python
arr + 10          # Add scalar
arr * 2           # Multiply
arr ** 2          # Power
np.sqrt(arr)      # Square root
```

**Statistical**:
```python
np.mean(arr)      # Mean
np.std(arr)       # Standard deviation
np.min(arr)       # Minimum
np.max(arr)       # Maximum
np.sum(arr)       # Sum
```

#### 6. Broadcasting

**Definition**: Automatic dimension expansion for operations

**Rules**:
1. Dimensions compared from right to left
2. Dimensions must be equal or one must be 1
3. Missing dimensions treated as 1

**Example**:
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
arr + 10                                  # Scalar broadcasts
arr + np.array([10, 20, 30])             # (3,) broadcasts to (2, 3)
```

#### 7. Linear Algebra

**Dot Product**:
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.dot(a, b)      # 32
a @ b             # Alternative syntax
```

**Matrix Multiplication**:
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
A @ B             # Matrix multiplication
A * B             # Element-wise (different!)
```

---

## Pandas for Data Science

### What is Pandas?

**Pandas** provides DataFrames for data manipulation and analysis.

### Key Data Structures

#### 1. Series

**Definition**: 1-dimensional labeled array

**Properties**:
- Has index
- Like a column in Excel
- Can hold any data type

**Example**:
```python
s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
```

#### 2. DataFrame

**Definition**: 2-dimensional labeled data structure

**Properties**:
- Rows and columns
- Like Excel spreadsheet
- Most commonly used

**Creation**:
```python
df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
```

### Essential Operations

#### 1. Reading Data

```python
# CSV
df = pd.read_csv('file.csv')

# Excel
df = pd.read_excel('file.xlsx')

# JSON
df = pd.read_json('file.json')

# SQL
df = pd.read_sql('SELECT * FROM table', connection)
```

#### 2. Data Exploration

```python
df.head()          # First 5 rows
df.tail()          # Last 5 rows
df.info()          # Data types and memory
df.describe()      # Statistical summary
df.shape           # (rows, columns)
df.columns         # Column names
df.dtypes          # Data types
```

#### 3. Selecting Data

```python
# Columns
df['col']          # Single column
df[['col1', 'col2']]  # Multiple columns

# Rows
df.iloc[0]         # By position
df.loc[0]         # By label
df[df['col'] > 5]  # Filtering
```

#### 4. Data Manipulation

```python
# Adding columns
df['new_col'] = df['col1'] * 2

# Dropping
df.drop('col', axis=1)      # Drop column
df.dropna()                 # Drop missing

# Sorting
df.sort_values('col')       # Sort by column

# Grouping
df.groupby('col').mean()    # Group and aggregate
```

#### 5. Handling Missing Data

**Detection**:
```python
df.isnull()        # Boolean mask
df.isnull().sum()  # Count missing per column
```

**Handling**:
```python
df.dropna()                    # Remove rows with missing
df.fillna(0)                   # Fill with value
df.fillna(df.mean())           # Fill with mean
df['col'].fillna(df['col'].median())  # Fill specific column
```

#### 6. Categorical Encoding

**One-Hot Encoding**:
```python
pd.get_dummies(df, columns=['category'])
```

**Label Encoding**:
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['encoded'] = le.fit_transform(df['category'])
```

---

## Data Visualization

### Why Visualize?

- **Understand Patterns**: See trends and relationships
- **Detect Outliers**: Identify anomalies
- **Communicate**: Share findings effectively
- **Validate**: Check assumptions

### Types of Visualizations

#### 1. Univariate (Single Variable)

**Histogram**:
- Shows distribution
- Use for: Continuous data
- Reveals: Shape, center, spread

**Box Plot**:
- Shows distribution and outliers
- Use for: Comparing groups
- Reveals: Median, quartiles, outliers

**Bar Chart**:
- Shows counts
- Use for: Categorical data
- Reveals: Frequency

#### 2. Bivariate (Two Variables)

**Scatter Plot**:
- Shows relationship
- Use for: Two continuous variables
- Reveals: Correlation, patterns

**Line Plot**:
- Shows trends
- Use for: Time series
- Reveals: Changes over time

**Heatmap**:
- Shows correlation matrix
- Use for: Multiple variables
- Reveals: Relationships

#### 3. Multivariate (Multiple Variables)

**Pair Plot**:
- All pairwise relationships
- Use for: Comprehensive EDA
- Reveals: Overall patterns

**Faceted Plots**:
- Multiple subplots
- Use for: Comparing groups
- Reveals: Group differences

### Best Practices

1. **Clear Labels**: Always label axes
2. **Meaningful Titles**: Descriptive titles
3. **Appropriate Colors**: Use colorblind-friendly palettes
4. **Right Chart Type**: Match data to visualization
5. **Avoid Clutter**: Keep it simple
6. **Consistent Style**: Use same style throughout

---

## Data Preprocessing

### Why Preprocess?

- **ML Requirements**: Models need clean, formatted data
- **Performance**: Better preprocessing = better models
- **Accuracy**: Handle missing values, outliers

### Preprocessing Pipeline

#### 1. Data Cleaning

**Steps**:
- Remove duplicates
- Handle missing values
- Fix data types
- Remove outliers (if appropriate)

#### 2. Feature Engineering

**Creating Features**:
```python
# From existing features
df['ratio'] = df['col1'] / df['col2']
df['interaction'] = df['col1'] * df['col2']

# From dates
df['year'] = pd.to_datetime(df['date']).dt.year
df['month'] = pd.to_datetime(df['date']).dt.month
```

#### 3. Encoding Categorical Variables

**One-Hot Encoding**:
- Creates binary columns
- Use for: Nominal categories
- Example: City → city_NYC, city_LA, city_Chicago

**Label Encoding**:
- Assigns numbers
- Use for: Ordinal categories
- Example: Small=1, Medium=2, Large=3

#### 4. Scaling/Normalization

**Standardization (Z-score)**:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled = scaler.fit_transform(data)
# Mean=0, Std=1
```

**Min-Max Scaling**:
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)
# Range: 0 to 1
```

**Why Scale?**:
- Some algorithms sensitive to scale (SVM, KNN, Neural Networks)
- Features on different scales can bias model

#### 5. Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Why Split?**:
- Train on training set
- Evaluate on test set (unseen data)
- Prevents overfitting assessment

### Complete Preprocessing Example

```python
# 1. Load data
df = pd.read_csv('data.csv')

# 2. Explore
print(df.info())
print(df.describe())

# 3. Handle missing
df = df.fillna(df.median())

# 4. Encode categorical
df = pd.get_dummies(df, columns=['category'])

# 5. Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['feature1', 'feature2']] = scaler.fit_transform(df[['feature1', 'feature2']])

# 6. Split
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

---

## Key Concepts Summary

### Python for ML

- **Lists**: For general data
- **Dictionaries**: For structured data
- **List Comprehensions**: Efficient data transformation
- **Functions**: Reusable code blocks

### NumPy for ML

- **Arrays**: Fast numerical operations
- **Vectorization**: Operations on entire arrays
- **Broadcasting**: Automatic dimension expansion
- **Linear Algebra**: Matrix operations

### Pandas for ML

- **DataFrames**: Main data structure
- **Data Cleaning**: Handle missing values
- **Feature Engineering**: Create new features
- **Data Manipulation**: Filter, group, merge

### Visualization for ML

- **EDA**: Explore before modeling
- **Pattern Detection**: Find relationships
- **Communication**: Share findings
- **Validation**: Check assumptions

---

## Best Practices

1. **Always Explore Data First**: Use head(), describe(), visualizations
2. **Handle Missing Values**: Before modeling
3. **Encode Categorical Variables**: Models need numbers
4. **Scale Features**: For algorithms that need it
5. **Split Data**: Train/test split before preprocessing
6. **Document**: Comment your code
7. **Version Control**: Use Git for your projects

---

## Common Mistakes to Avoid

1. ❌ Not exploring data first
2. ❌ Ignoring missing values
3. ❌ Not encoding categorical variables
4. ❌ Forgetting to scale features
5. ❌ Data leakage (preprocessing before split)
6. ❌ Over-complicating preprocessing
7. ❌ Not documenting steps

---

## Next Steps

After mastering foundations:

1. ✅ Move to ML algorithms
2. ✅ Build your first model
3. ✅ Practice with real datasets
4. ✅ Learn model evaluation
5. ✅ Explore advanced topics

---

**Remember**: Strong foundations lead to better ML models. Master these basics before moving forward!

