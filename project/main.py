import model
import data
import util

import numpy as np

text = """Hey! I rewrote your C++ code:

int main() {
    std::vector<Person> people = {{""Alice"", 30}, {""Bob"", 25}, {""Charlie"", 35}};
    std::sort(people.begin(), people.end(), compareByAge);
    for (const auto &person : people) {
        std::cout << person.name << "" is "" << person.age << "" years old."" << std::endl;
    }
    return 0;
}

on Python as you asked:

def main():
    people = [Person(""Alice"", 30), Person(""Bob"", 25), Person(""Charlie"", 35)]
    people.sort(key=compare_by_age)
    for person in people:
        print(f""{person.name} is {person.age} years old."")


hope it helps!
"""

class Controller:

    def __init__(self):

        self.text_classifier = model.TextClassifier()
        self.code_classifier = model.CodeClassifier()

        self.text_classifier.load()
        self.code_classifier.load()

    def process(self, text):

        segments = []

        for segment in data.extract(text):

            if self.text_classifier.topics[np.argmax(self.text_classifier.predict(segment))] == 'code':

                segments.append(segment)

        for segment in segments:

            label = self.code_classifier.topics[np.argmax(self.code_classifier.predict(segment))]

            print(f"Code segment, {label}\n{segment}\n\n")
