def main() -> None:
    print("Hello World from uv!")

if __name__ == "__main__":
    main()

name: str = "Alice"
age: int = 25
height: float = 164.5
is_Active: bool = True
age: str = "Here"

print(f"User {name} is {age} years old and is {height}cm tall")

def addToAge(years:int) -> int:
    return age 

from typing import Optional


def greet(name: str, title: Optional[str] = None) -> str:
    if title:
        return f"Hello {title} {name}"
    return f"Hello {name}"       

print(addToAge(2))
print(greet("Rafael"))
print(greet("Rafael", "Mr"))

numbers: list[int] = [1,2,3,4,5]
scores: dict[str, int] = {"Alice": 12, "Ted": 31}

evens_time_ten = [n * 10 for n in numbers if n %2 == 0]
print(evens_time_ten)

class Person:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
    def celebrate_birthday(self) -> None:
        self.age += 1
        print(f"Happy Birthday {self.name}, you are now {self.age} yo")

p = Person("Rafa", 31)
p.celebrate_birthday()

def here_we_go(name:str) -> str:
    numbers: int = [1,2,4]
    myNum: int = [n for n in numbers if n < 0]
    return f"Here we go {name}, we are ready, your number is {myNum}"

def test(age:int) -> int:
    return 100

print(here_we_go("Joe Rogan"))

print(test(19))

score: int = 107 

if score > 90:
    print("greater that 90")
elif score > 60:
    print("greater than 60")
elif score > 40:
    print ("greater than 40")
else:
    print("rapaz, sei nao")

