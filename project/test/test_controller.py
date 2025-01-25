import pytest
import numpy
import sys
sys.path.append('.')

import util

@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("query, language", [
    (
"""Hey! I rewrote your C++ code:

int main() {
    std::vector<Person> people = {{"Alice", 30}, {"Bob", 25}, {"Charlie", 35}};
    std::sort(people.begin(), people.end(), compareByAge);
    for (const auto &person : people) {
        std::cout << person.name << \" is \" << person.age << \" years old.\" << std::endl;
    }
    return 0;
}

on Python as you asked:

def main():
    people = [Person(\"Alice\", 30), Person(\"Bob\", 25), Person(\"Charlie\", 35)]
    people.sort(key=compare_by_age)
    for person in people:
        print(f\"{person.name} is {person.age} years old.\")

hope it helps!
""", "Code segment, c + +"),
    (
"""Hey! I rewrote your C++ code:

int main() {
    std::vector<Person> people = {{"Alice", 30}, {"Bob", 25}, {"Charlie", 35}};
    std::sort(people.begin(), people.end(), compareByAge);
    for (const auto &person : people) {
        std::cout << person.name << \" is \" << person.age << \" years old.\" << std::endl;
    }
    return 0;
}

on Python as you asked:

def main():
    people = [Person(\"Alice\", 30), Person(\"Bob\", 25), Person(\"Charlie\", 35)]
    people.sort(key=compare_by_age)
    for person in people:
        print(f\"{person.name} is {person.age} years old.\")

hope it helps!
""", "Code segment, python")
])
def test_Controller(query, language, capfd):

    controller = util.Controller()
    controller.process(query)

    out, err = capfd.readouterr()

    assert language in out
