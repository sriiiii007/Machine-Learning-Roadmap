import numpy as np
import matplotlib.pyplot as plt

print("+"*70)
print("Numpy hands-on practice")

print("Numpy is mainly used for numberical computing in python,\
  this is the foundation for the other libraries like pandas, Scikit-learn,\
  TensorFlow, PyTorch and many more.\
  Main things we should understand is the concept of Broadcatsing which means the \
  operations on arrays of different shapes and vectorization is the operation on entire \
  array at once. Broadcatsing can also be said as teh automatic expansion of the arrays \
  dimension in the other words.\n Array is always a homogenous collection of the elemnets \n Mainly \
  used for statistical modeling like mean, mode, avg, std, var, correlation etc and also \
  predominently used on linear algebra and array manipulation like contatinating, reshape, transpose etc ")

print("\n Practical implementation begins...")

python_list = [10,20,30,40,50,60]
array = np.array(python_list)
print(f"python list: {python_list}")
print(f"python numpy array: {array}")
print(f"Type: {type(python_list)}")
print(f"Type: {type(array)}")
print(f"List Shape: {len(python_list)}")
print(f"Shape: {array.shape}")
print(f"Dtype: {array.dtype}")

##Multi-dimensional arrays
matrix = np.array([[1,2],[5,6],[8,9]])
print(f"Matrix shape: {matrix.shape}")
print(f"Matrix : {matrix}")
print(f"Dimensions:{matrix.ndim}")
print(f"Dtypes :{matrix.dtype}")

##Special types
zeros = np.zeros((2,3))
print(zeros)

ones = np.ones((4,5))
print(ones)

fullMatrix = np.full((3,3),2)
print(fullMatrix)

identity = np.eye(5)
print(identity)

random_arr = np.random.rand(5,5)
print(random_arr.shape)
print(random_arr)

print("Creating arrays with ranges :")
arange = np.arange(0,20,2)
linspace = np.linspace(0,10,10)

print(arange)
print(linspace)

###Array operations

array1 = np.array([5,6,7,9])
array2 = np.array([11,12,13,14])
print(f" Addition of arrays : {array1+array2}")
print(f" Subtraction of arrays : {array1-array2}")
print(f" Multiplication of arrays : {array1*array2}")
print(f" Division of arrays : {array1/array2}")
print(f"Power of arrays: {array1 ** array2}")
print(f"Square root of arrays:{np.sqrt(array2)}")
print(f"Sine of the array: {np.sin(array1)}")

###Statistical operations
data = np.array([3,5,7,9,11,13])
print(f"Mean:{np.mean(data)}")
print(f"Median:{np.median(data)}")
print(f"Standard deviation:{np.std(data)}")
print(f"Variance:{np.var(data)}")
print(f"Minimum:{np.min(data)}")
print(f"Maximum:{np.max(data)}")
print(f"Sum: {np.sum(data)}")
print(f"Product:{np.prod(data)}")
print(f"Length:{np.size(data)}")
print(f"Average calulation :{np.sum(data)/np.size(data)}")

### Array indexing and slicing
arr = np.array([[2,4,6],[8, 10, 12],[14, 16, 18]])
print(arr)

print(f"First row: {arr[0]}")
print(f"Second row: {arr[1]}")
print(f"Third row: {arr[2,0]}")
print(f"Last element: {arr[-1, -2]}")

print("--------Slicing--------")
print(f"arr:{arr[0, 1:3]}")
print(f"arr[2, 0:2]:{arr[2, 0:2]}")
print(f"arr[:, 0] = {arr[:, 0]}")
print(f"arr[0:2, 1:3] =\n{arr[0:2, 1:3]}")
arr_1d = np.arange(13)
print(arr_1d)

print("Boolean Indexing:")
mask = arr > 6
print(mask)
print(f"To print the actual matrix values greater than the mask:{arr[mask]}")
