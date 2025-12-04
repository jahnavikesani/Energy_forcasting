path = r'C:\Users\jahna\OneDrive\Desktop\overhead\energy-forecast\backend\app\main.py'

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Remove markdown code fences
content = content.replace('````python\n', '')
content = content.replace('````\n', '')
content = content.replace('```python\n', '')
content = content.replace('```\n', '')
content = content.replace('````', '')
content = content.replace('```', '')

with open(path, 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)

print('Removed all backticks from file')
