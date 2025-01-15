import re

keywords = {
    'python': [
        'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 
        'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 
        'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield', 'print'
    ],
    'cpp': [
        'alignas', 'alignof', 'and', 'and_eq', 'asm', 'atomic_cancel', 'atomic_commit', 'atomic_noexcept', 'auto', 
        'bitand', 'bitor', 'bool', 'break', 'case', 'catch', 'char', 'char8_t', 'char16_t', 'char32_t', 'class', 
        'compl', 'concept', 'const', 'consteval', 'constexpr', 'constinit', 'const_cast', 'continue', 'co_await', 
        'co_return', 'co_yield', 'decltype', 'default', 'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum', 
        'explicit', 'export', 'extern', 'false', 'float', 'for', 'friend', 'goto', 'if', 'inline', 'int', 'long', 
        'mutable', 'namespace', 'new', 'noexcept', 'not', 'not_eq', 'nullptr', 'operator', 'or', 'or_eq', 'private', 
        'protected', 'public', 'reflexpr', 'register', 'reinterpret_cast', 'requires', 'return', 'short', 'signed', 
        'sizeof', 'static', 'static_assert', 'static_cast', 'struct', 'switch', 'synchronized', 'template', 'this', 
        'thread_local', 'throw', 'true', 'try', 'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using', 
        'virtual', 'void', 'volatile', 'wchar_t', 'while', 'xor', 'xor_eq', 'cout', 'cin'
    ],
    'java': [
        'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', 'const', 'continue', 
        'default', 'do', 'double', 'else', 'enum', 'extends', 'final', 'finally', 'float', 'for', 'goto', 'if', 
        'implements', 'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new', 'null', 'package', 
        'private', 'protected', 'public', 'return', 'short', 'static', 'strictfp', 'super', 'switch', 'synchronized', 
        'this', 'throw', 'throws', 'transient', 'try', 'void', 'volatile', 'while'
    ]
}

def extract(text):

    """
    Extract code blocks from a given text based on programming language keywords.

    Parameters:
        text - input text containing potential code blocks.

    Return:
        code - list of strings, each representing a block of code found in the input text.
    """

    pattern = r'^\s*(' + '|'.join(keywords['python'] + keywords['cpp'] + keywords['java']) + r')\b|^\s*\w+\s*=|^\s*\w+\s*\(.*\)\s*|^\s*\w+\s*\(.*=.*\)\s*'
    pattern_compiled = re.compile(pattern, re.MULTILINE)

    code = []
    current_code = []
    flag = False

    for line in text.split('\n'):

        if pattern_compiled.match(line):

            if not flag:

                flag = True
                current_code = [line]

            else:

                current_code.append(line)

        else:

            if flag:

                code.append('\n'.join(current_code))
                flag = False

            current_code = []

    if flag:

        code.append('\n'.join(current_code))

    return code
