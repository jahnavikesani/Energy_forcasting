import re

path = r'C:\Users\jahna\OneDrive\Desktop\overhead\energy-forecast\backend\app\main.py'

# Read the file
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace unicode characters
content = re.sub(r'[\u2018\u2019]', "'", content)  # Smart single quotes
content = re.sub(r'[\u201C\u201D]', '"', content)  # Smart double quotes
content = re.sub(r'[\u2013\u2014]', '-', content)  # En/em dashes
content = re.sub(r'\u20AC', '', content)           # Euro symbol
content = re.sub(r'\u2026', '...', content)        # Ellipsis

# Write back
with open(path, 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)

print('Fixed all unicode characters in main.py')
