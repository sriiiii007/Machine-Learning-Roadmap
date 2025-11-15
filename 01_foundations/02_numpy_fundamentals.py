"""
NumPy Fundamentals for Machine Learning
Complete guide to NumPy - the foundation of all ML libraries.
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("NUMPY FUNDAMENTALS - Theory & Practice")
print("=" * 70)

# ============================================================================
# THEORY: What is NumPy?
# ============================================================================
print("\n" + "=" * 70)
print("THEORY: Understanding NumPy")
print("=" * 70)

theory = """
NUMPY (Numerical Python)

1. WHAT IS NUMPY?
   - Core library for numerical computing in Python
   - Foundation for Pandas, Scikit-learn, TensorFlow, PyTorch
   - Provides N-dimensional arrays (ndarray)
   - Much faster than Python lists

2. WHY NUMPY?
   - Speed: Written in C, optimized for performance
   - Memory: Efficient storage
   - Broadcasting: Operations on arrays of different sizes
   - Vectorization: Operations on entire arrays at once

3. KEY CONCEPTS:
   - Array: Homogeneous collection of elements
   - Shape: Dimensions of array (rows, columns, ...)
   - Dtype: Data type (int, float, bool, etc.)
   - Broadcasting: Automatic dimension expansion

4. COMMON OPERATIONS:
   - Mathematical operations (+, -, *, /, **)
   - Statistical functions (mean, std, min, max)
   - Linear algebra (dot product, matrix multiplication)
   - Array manipulation (reshape, transpose, concatenate)
"""

print(theory)

# ============================================================================
# PRACTICE: Creating Arrays
# ============================================================================
print("\n" + "=" * 70)
print("PRACTICE: Creating NumPy Arrays")
print("=" * 70)

# 1. From Python lists
print("\n1. Creating arrays from Python lists:")
python_list = [1, 2, 3, 4, 5]
arr1 = np.array(python_list)
print(f"Python list: {python_list}")
print(f"NumPy array: {arr1}")
print(f"Type: {type(arr1)}")
print(f"Shape: {arr1.shape}")
print(f"Dtype: {arr1.dtype}")

# 2. Multi-dimensional arrays
print("\n2. Creating 2D arrays:")
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"2D Array:\n{matrix}")
print(f"Shape: {matrix.shape}")  # (3, 3) = 3 rows, 3 columns
print(f"Dimensions: {matrix.ndim}")

# 3. Special arrays
print("\n3. Creating special arrays:")
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
full = np.full((2, 2), 7)
identity = np.eye(3)
random_arr = np.random.rand(3, 3)

print(f"Zeros (3x4):\n{zeros}")
print(f"\nOnes (2x3):\n{ones}")
print(f"\nFull of 7s (2x2):\n{full}")
print(f"\nIdentity (3x3):\n{identity}")
print(f"\nRandom (3x3):\n{random_arr}")

# 4. Arrays with ranges
print("\n4. Creating arrays with ranges:")
arange = np.arange(0, 10, 2)  # Start, stop, step
linspace = np.linspace(0, 1, 5)  # Start, stop, num_points

print(f"arange(0, 10, 2): {arange}")
print(f"linspace(0, 1, 5): {linspace}")

# ============================================================================
# ARRAY OPERATIONS
# ============================================================================
print("\n" + "=" * 70)
print("Array Operations")
print("=" * 70)

# Mathematical operations
print("\n1. Mathematical Operations:")
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print(f"a = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")
print(f"a * 2 = {a * 2}")
print(f"a ** 2 = {a ** 2}")
print(f"np.sqrt(a) = {np.sqrt(a)}")
print(f"np.sin(a) = {np.sin(a)}")

# Statistical operations
print("\n2. Statistical Operations:")
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"Data: {data}")
print(f"Mean: {np.mean(data)}")
print(f"Median: {np.median(data)}")
print(f"Std Dev: {np.std(data)}")
print(f"Min: {np.min(data)}")
print(f"Max: {np.max(data)}")
print(f"Sum: {np.sum(data)}")
print(f"Product: {np.prod(data)}")

# ============================================================================
# ARRAY INDEXING AND SLICING
# ============================================================================
print("\n" + "=" * 70)
print("Array Indexing and Slicing")
print("=" * 70)

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(f"Array:\n{arr}")

print("\n1. Basic Indexing:")
print(f"arr[0, 0] = {arr[0, 0]}")  # First element
print(f"arr[1, 2] = {arr[1, 2]}")  # Row 1, Column 2
print(f"arr[-1, -1] = {arr[-1, -1]}")  # Last element

print("\n2. Slicing:")
print(f"arr[0, :] = {arr[0, :]}")  # First row
print(f"arr[:, 0] = {arr[:, 0]}")  # First column
print(f"arr[0:2, 1:3] =\n{arr[0:2, 1:3]}")  # Submatrix

print("\n3. Boolean Indexing:")
mask = arr > 5
print(f"Mask (arr > 5):\n{mask}")
print(f"Values > 5: {arr[mask]}")

# ============================================================================
# ARRAY MANIPULATION
# ============================================================================
print("\n" + "=" * 70)
print("Array Manipulation")
print("=" * 70)

# Reshape
print("\n1. Reshaping:")
arr_1d = np.arange(12)
arr_2d = arr_1d.reshape(3, 4)
print(f"1D array: {arr_1d}")
print(f"Reshaped to 3x4:\n{arr_2d}")

# Transpose
print("\n2. Transpose:")
print(f"Original:\n{arr_2d}")
print(f"Transposed:\n{arr_2d.T}")

# Flatten
print("\n3. Flatten:")
print(f"Flattened: {arr_2d.flatten()}")

# Concatenate
print("\n4. Concatenation:")
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print(f"Array a:\n{a}")
print(f"Array b:\n{b}")
print(f"Vertical concatenation:\n{np.vstack((a, b))}")
print(f"Horizontal concatenation:\n{np.hstack((a, b))}")

# ============================================================================
# LINEAR ALGEBRA (Important for ML)
# ============================================================================
print("\n" + "=" * 70)
print("Linear Algebra Operations (Critical for ML)")
print("=" * 70)

# Dot product
print("\n1. Dot Product:")
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot_product = np.dot(a, b)
print(f"a = {a}")
print(f"b = {b}")
print(f"Dot product: {np.dot(a, b)}")
print(f"Alternative: {a @ b}")

# Matrix multiplication
print("\n2. Matrix Multiplication:")
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"Matrix A:\n{A}")
print(f"Matrix B:\n{B}")
print(f"A @ B:\n{A @ B}")

# Element-wise multiplication
print("\n3. Element-wise Multiplication:")
print(f"A * B:\n{A * B}")

# ============================================================================
# PRACTICAL ML EXAMPLE: Data Preprocessing
# ============================================================================
print("\n" + "=" * 70)
print("Practical Example: Data Preprocessing for ML")
print("=" * 70)

# Simulate dataset
np.random.seed(42)
data = np.random.randn(100, 4)  # 100 samples, 4 features
print(f"Original data shape: {data.shape}")
print(f"First 5 rows:\n{data[:5]}")

# Normalization (Z-score)
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
normalized = (data - mean) / std
print(f"\nNormalized data (first 5 rows):\n{normalized[:5]}")

# Min-Max scaling
min_val = np.min(data, axis=0)
max_val = np.max(data, axis=0)
scaled = (data - min_val) / (max_val - min_val)
print(f"\nMin-Max scaled data (first 5 rows):\n{scaled[:5]}")

# ============================================================================
# BROADCASTING (Important Concept)
# ============================================================================
print("\n" + "=" * 70)
print("Broadcasting (Automatic Dimension Expansion)")
print("=" * 70)

print("""
Broadcasting allows operations between arrays of different shapes.

Rules:
1. Dimensions are compared from right to left
2. Dimensions must be equal or one must be 1
3. Missing dimensions are treated as 1

Examples:
""")

# Example 1: Adding scalar to array
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Array:\n{arr}")
print(f"Array + 10:\n{arr + 10}")

# Example 2: Adding 1D to 2D
row = np.array([10, 20, 30])
print(f"\nRow vector: {row}")
print(f"Array + row:\n{arr + row}")

# Example 3: Adding column
col = np.array([[10], [20]])
print(f"\nColumn vector:\n{col}")
print(f"Array + col:\n{arr + col}")

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================
print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
1. NumPy arrays are faster and more efficient than Python lists
2. Vectorization: Operations on entire arrays (no loops needed)
3. Broadcasting: Automatic dimension expansion
4. Essential for ML: All ML libraries use NumPy arrays
5. Practice: Try operations on your own arrays

COMMON NUMPY FUNCTIONS FOR ML:
- np.array(): Create arrays
- np.zeros(), np.ones(): Special arrays
- np.random.rand(): Random arrays
- np.mean(), np.std(): Statistics
- np.dot(), @: Matrix operations
- np.reshape(): Change shape
- np.concatenate(): Combine arrays

NEXT STEPS:
- Practice with different array operations
- Try reshaping and manipulating arrays
- Move to Pandas for data manipulation
"""

print(takeaways)

print("\n" + "=" * 70)
print("Next: Learn Pandas for data manipulation")
print("=" * 70)

