"""Fix indentation in main.py"""

path = r'C:\Users\jahna\OneDrive\Desktop\overhead\energy-forecast\backend\app\main.py'

with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed_lines = []
indent_level = 0

for i, line in enumerate(lines):
    stripped = line.lstrip()
    
    # Skip empty lines
    if not stripped:
        fixed_lines.append('\n')
        continue
    
    # Dedent for except, finally, elif, else
    if stripped.startswith(('except ', 'except:', 'finally:', 'elif ', 'else:')):
        indent_level = max(0, indent_level - 1)
    
    # Apply current indentation
    fixed_line = '    ' * indent_level + stripped
    fixed_lines.append(fixed_line)
    
    # Indent after colons (for def, class, if, for, while, try, except, etc.)
    if stripped.rstrip().endswith(':'):
        if not stripped.startswith(('"""', "'''", '#')):  # Ignore docstrings and comments
            indent_level += 1
    
    # Dedent after return, pass, break, continue, raise (if standalone)
    if stripped.startswith(('return', 'pass', 'break', 'continue', 'raise')) and not stripped.rstrip().endswith(':'):
        # Check if next line exists and doesn't start with more indented code
        if i + 1 < len(lines):
            next_stripped = lines[i + 1].lstrip()
            # If next line is dedented or is a dedenting keyword, reduce indent
            if next_stripped and not next_stripped.startswith(('"""', "'''", '#')):
                if next_stripped.startswith(('def ', 'class ', 'except ', 'except:', 'finally:', 'elif ', 'else:', '@')):
                    indent_level = max(0, indent_level - 1)

with open(path, 'w', encoding='utf-8', newline='\n') as f:
    f.writelines(fixed_lines)

print(f'Fixed indentation for {len(fixed_lines)} lines')
