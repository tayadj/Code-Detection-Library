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

text_ = """Hey, here's fibonacci algorithm on different languages:

def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    else:
        fibs = [0, 1]
        for i in range(2, n):
            fibs.append(fibs[-1] + fibs[-2])
        return fibs

that above is on python, next is gonna be on swift

func fibonacci(_ n: Int) -> [Int] {
    if n <= 0 {
        return []
    } else if n == 1 {
        return [0]
    } else if n == 2 {
        return [0, 1]
    } else {
        var fibs = [0, 1]
        for i in 2..<n {
            fibs.append(fibs[i-1] + fibs[i-2])
        }
        return fibs
    }
}

and then on c++

vector<int> fibonacci(int n) {
    vector<int> fibs;
    if (n <= 0) return fibs;
    fibs.push_back(0);
    if (n == 1) return fibs;
    fibs.push_back(1);
    for (int i = 2; i < n; ++i) {
        fibs.push_back(fibs[i-1] + fibs[i-2]);
    }
    return fibs;
}

import java.util.ArrayList;
import java.util.List;
public class Fibonacci {
    public static List<Integer> fibonacci(int n) {
        List<Integer> fibs = new ArrayList<>();
        if (n <= 0) return fibs;
        fibs.add(0);
        if (n == 1) return fibs;
        fibs.add(1);
        for (int i = 2; i < n; ++i) {
            fibs.add(fibs.get(i-1) + fibs.get(i-2));
        }
        return fibs;
    }
}

and the last one on java
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
