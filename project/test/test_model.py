import pytest
import numpy
import sys
sys.path.append('.')

import model



@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("query, prediction", [
    ("Hey, here's your code", "message"),
    ("std::cout << \"Hello, world!\";", "code"),
    ("def my_function():", "code"),
    ("Can you help me to find an error?", "message"),
    ("How are you today?", "message"),
    ("Let's meet at 5 PM.", "message"),
    ("class MyClass:", "code")
])
def test_TextClassifier(query, prediction):

    classifier = model.TextClassifier()
    classifier.load()

    assert classifier.topics[numpy.argmax(classifier.predict(query))] == prediction



@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("query, prediction", [
    ("std::cout << \"Hello, world!\";", "c + +"),
    ("print('Hello, world!')", "python"),
    ("def my_function():", "python"),
    ("System.out.println(\"Hello, world!\");", "java"),
    ("public class HelloWorld { public static void main(String[] args) { System.out.println(\"Hello, world!\"); } }", "java"),
    ("#include <iostream>\nint main() { std::cout << \"Hello, world!\" << std::endl; return 0; }", "c + +"),
    ("let greeting = \"Hello, world!\"\nprint(greeting)", "swift"),
    ("for i in range(5): print(f\"i = {i}\")", "python"),
    ("public class Animal { public void speak() { System.out.println(\"Animal speaks\"); } }", "java"),
    ("class Animal:\n    def speak(self):\n        print(\"Animal speaks\")", "python"),
    ("public class ForLoop { public static void main(String[] args) { for (int i = 0; i < 5; i++) { System.out.println(\"i = \" + i); } } }", "java")
])
def test_CodeClassifier(query, prediction):

    classifier = model.CodeClassifier()
    classifier.load()

    assert classifier.topics[numpy.argmax(classifier.predict(query))] == prediction
