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
