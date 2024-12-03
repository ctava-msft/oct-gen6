import sys
import os
import re

def convert_matlab_to_python(matlab_code):
    python_code = matlab_code

    # Replace comments
    python_code = re.sub(r'%{[^%]*%}', lambda m: '\n'.join(['# ' + line for line in m.group(0).splitlines()]), python_code)
    python_code = re.sub(r'%.*', lambda m: '#' + m.group(0)[1:], python_code)

    # Replace function definitions
    python_code = re.sub(r'function\s*(\[[^\]]*\]|[\w]+)\s*=\s*([\w]+)\s*\((.*?)\)', r'def \2(\3):', python_code)

    # Replace end statements
    python_code = re.sub(r'\bend\b', '', python_code)

    # Replace if statements
    python_code = re.sub(r'\bif\s+(.*)', r'if \1:', python_code)
    python_code = re.sub(r'\belseif\s+(.*)', r'elif \1:', python_code)
    python_code = re.sub(r'\belse\b', 'else:', python_code)

    # Replace for loops
    python_code = re.sub(r'\bfor\s+(\w+)\s*=\s*(.+)', r'for \1 in \2:', python_code)

    # Replace while loops
    python_code = re.sub(r'\bwhile\s+(.*)', r'while \1:', python_code)

    # Replace logical operators
    python_code = python_code.replace('~=', '!=')
    python_code = python_code.replace('~', 'not ')
    python_code = python_code.replace('&', 'and')
    python_code = python_code.replace('|', 'or')

    # Replace array indexing
    python_code = re.sub(r'(\w+)\((.+?)\)', r'\1[\2]', python_code)

    # Replace transpose operator
    python_code = python_code.replace(".'", ".T")

    return python_code

def main():
    if len(sys.argv) != 2:
        print("Usage: python convert_matlab_to_python.py <input_file.m>")
        sys.exit(1)

    input_file = sys.argv[1]

    if not os.path.isfile(input_file):
        print(f"File {input_file} does not exist.")
        sys.exit(1)

    with open(input_file, 'r') as f:
        matlab_code = f.read()

    python_code = convert_matlab_to_python(matlab_code)

    output_file = os.path.splitext(input_file)[0] + '.py'
    with open(output_file, 'w') as f:
        f.write(python_code)

    print(f"Converted {input_file} to {output_file}")

if __name__ == "__main__":
    main()