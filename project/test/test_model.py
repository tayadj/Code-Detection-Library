from model import Base, TextClassifier, CodeClassifier
import pytest
import numpy



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

    classifier = TextClassifier()
    classifier.load()

    assert classifier.topics[numpy.argmax(classifier.predict(query))] == prediction



@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("query, prediction", [
    ("std::cout << \"Hello, world!\";", "c + +"),
    ("print('Hello, world!')", "python"),
    ("def my_function():", "python"),
])
def test_CodeClassifier(query, prediction):

    classifier = CodeClassifier()
    classifier.load()

    assert classifier.topics[numpy.argmax(classifier.predict(query))] == prediction
