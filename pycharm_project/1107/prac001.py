class TestClass:
    pass

object1 = TestClass()
object2 = TestClass()

print("object1: ", type(object1))
print("object2: ", type(object2))

class Person:
    def say_hello(self, name):
        print('Hello!',name)
    def say_bye(self, name):
        print('Goodbye!', name)


person = Person()
person.say_hello()