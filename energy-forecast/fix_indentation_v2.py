"""Fix indentation in main.py by parsing Python structure"""
import re
import tokenize
import io

path = r'C:\Users\jahna\OneDrive\Desktop\overhead\energy-forecast\backend\app\main.py'

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# First pass: identify lines that should be indented
lines = content.split('\n')
fixed_lines = []
indent_stack = [0]  # Stack to track indentation levels

for i, line in enumerate(lines):
    stripped = line.lstrip()
    
    # Keep empty lines and comments as-is with proper indentation
    if not stripped or stripped.startswith('#'):
        if indent_stack:
            fixed_lines.append('    ' * (len(indent_stack) - 1) + stripped)
        else:
            fixed_lines.append(stripped)
        continue
    
    # Handle dedenting keywords
    if re.match(r'^(except|finally|elif|else)[\s:]', stripped):
        if len(indent_stack) > 1:
            indent_stack.pop()
    
    # Calculate current indentation
    current_indent = len(indent_stack) - 1
    fixed_line = '    ' * current_indent + stripped
    fixed_lines.append(fixed_line)
    
    # Handle indenting - check if line ends with colon
    if stripped.rstrip().endswith(':'):
        # Check it's not a string or comment
        if not (stripped.startswith(('"""', "'''", '#'))):
            indent_stack.append(current_indent + 1)
    
    # Handle dedenting after certain statements
    # Look ahead to see if we should dedent
    if i + 1 < len(lines):
        next_line = lines[i + 1].lstrip()
        if next_line:
            # If next line is a def/class at same or lower level, dedent
            if re.match(r'^(def|class|@|except|finally|elif|else)\s', next_line):
                # Pop back to appropriate level
                while len(indent_stack) > 1:
                    indent_stack.pop()
                    # Check if we should stop dedenting
                    if next_line.startswith(('except', 'finally', 'elif', 'else')):
                        if len(indent_stack) > 1:
                            break
                    else:
                        break

# Write the fixed content
with open(path, 'w', encoding='utf-8', newline='\n') as f:
    f.write('\n'.join(fixed_lines))

print(f'Fixed indentation for {len(fixed_lines)} lines')
