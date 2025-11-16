###Day1
name = "Srinidhi"
age = 26
height = 5.6
likes_ml = True

print(f"Name:{name}, Age:{age}, Height:{height}, like_ml:{likes_ml}")


numbers = [1, 2, 3, 4, 5]

info = {
    "first_name": "Sannidhi",
    "last_name": "MP",
    "age":25,
    "city":"Japan",
    "occuptaion":"Data Analyst",
    "friends":["A","C","D","E","F"]
}

print(f"Numbers:{numbers}")
print(f"First Name: {info['first_name']}")
print(f"Last Name: {info['last_name']}")
print(f"Friends:{info['friends'][4]}")

###Day2
for i in range(1,40): #thing to understand here is that the range function is exclusive of the last number, so we need to add 1 to the range to include the last number
    print(i, end=" ")
print("\n"+"="*50)

print(*range(1,20),sep=",") #this is a way to print the range without the comma separator

print("-".join(str(i) for i in range(1,10)))

for num in numbers:
    if num % 2 != 0:
        print(f"Odd number: {num}")

print("\n===Exercise 4:Functions===")

def cal_avg(num_list):
    if len(num_list) == 0:
        return 0
    print(f"\nSum of the numbers is : {sum(num_list)}")
    print(f"Length of the numbers is : {len(num_list)}")
    return  sum(num_list) / len(num_list)

print(f"\nAverage of numbers: {cal_avg(numbers)}")

###Example :
listToFindAVG = [100,200,300,400,500]
average = cal_avg(listToFindAVG)

print(f"Average of the list: {average}")

###Now to practice list comprehensions
cubes = [x*3 for x in range(1, 20)]
print(f"Cubes:{cubes}")

##So now the task is to find out the even numbers from the above Cubes list
evens = [x for x in cubes if x % 2==0]
print(f"Evens: {evens}")


###Done with practicing 01_python_basic.py for today, now moving to the next

###File 02_numpy_fundamentals.py
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
