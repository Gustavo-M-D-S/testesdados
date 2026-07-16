import re

with open('app/dashboard.py', 'r') as f:
    content = f.read()

old_pattern = 'dims = st.slider("Dimensions", 8, 128, 64, 8, key="emb_dims")'
new_code = '''n_nodes = G.number_of_nodes()
        max_dims = max(8, min(128, n_nodes - 1)) if n_nodes > 1 else 8
        dims = st.slider("Dimensions", 8, max_dims, min(64, max_dims), 8, key="emb_dims")'''

# This pattern should match the exact line with proper indentation
pattern = '        dims = st.slider("Dimensions", 8, 128, 64, 8, key="emb_dims")'
if pattern in content:
    content = content.replace(pattern, new_code)
    with open('app/dashboard.py', 'w') as f:
        f.write(content)
    print('SUCCESS: Fixed embedding dimensions slider')
else:
    print('Pattern not found - checking file...')
    # Show lines around line 777
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'slider' in line and 'emb_dims' in line:
            print(f'Line {i+1}: {repr(line)}')