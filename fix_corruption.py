import sys
import os

def fix_file(path):
    if not os.path.exists(path):
        print(f"Skipping {path} (not found)")
        return

    with open(path, 'r') as f:
        content = f.read()
    
    # Fix escaping
    # We assume the file has \" for " and \\ for \
    fixed = content.replace('\\"', '"')
    
    # We need to be careful with double backslashes. 
    # If the original code had `\n` inside a string, it became `\\n`.
    # Replacing `\\` with `\` restores it.
    fixed = fixed.replace('\\\\', '\\')
    
    with open(path, 'w') as f:
        f.write(fixed)
    print(f"Fixed {path}")

if __name__ == "__main__":
    files = [
        "prompt_prix/adapters/lmstudio.py",
        "prompt_prix/battery.py",
        "prompt_prix/mcp/tools/complete.py"
    ]
    for path in files:
        fix_file(path)
