class Person:
    def __init__(self, name):
        self.name = name
        self.say_hello()


    def say_hello(self):
        print("Hello! I'm", self.name)


person1 = Person("Kim")
person2 = Person("Yang")

