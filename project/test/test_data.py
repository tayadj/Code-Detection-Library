import pytest
import sys
sys.path.append('.')

import data

@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("message, code", [
    ("""
    Hey, I was working on that C++ function you mentioned. Here's what I came up with:

    #include <iostream>
    using namespace std;
    void greet(const string& name) {
        cout << "Hello, " << name << "!" << endl;
    }

    Let me know if you think it needs any changes!
    """, 
    """
    #include <iostream>
    using namespace std;
    void greet(const string& name) {
        cout << "Hello, " << name << "!" << endl;
    }
    """),
    ("""I guess I can code on Python, look:

    def greet(name):
        return f"Hello, {name}!"
    """,
    """
    def greet(name):
        return f"Hello, {name}!"
    """)
])
def test_extract(message, code):

    result = data.extract(message);
    result = [''.join(segment.split()) for segment in result]
    
    assert ''.join(code.split()) in result
