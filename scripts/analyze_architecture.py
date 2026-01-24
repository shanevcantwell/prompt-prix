#!/usr/bin/env python3
"""
Analyze the current prompt-prix architecture and generate documentation.
This script examines imports, class relationships, and module structure.
"""
import ast
import os
from pathlib import Path
from collections import defaultdict

def analyze_file(filepath):
    """Parse Python file and extract key information."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None
    
    info = {
        'imports': set(),
        'classes': [],
        'functions': [],
        'async_functions': []
    }
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                info['imports'].add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            if module and not module.startswith('__'):
                info['imports'].add(module)
        elif isinstance(node, ast.ClassDef):
            info['classes'].append(node.name)
        elif isinstance(node, ast.FunctionDef):
            if node.name != '__init__':
                func_type = 'async_functions' if any(
                    isinstance(d, ast.AsyncFor) or 
                    (isinstance(d, ast.Await) and not isinstance(d, ast.Call)) 
                    for d in ast.walk(node)
                ) else 'functions'
                info[func_type].append(node.name)
    
    return info

def main():
    base_path = Path('prompt_prix')
    modules = defaultdict(dict)
    
    # Analyze all Python files in prompt_prix
    for py_file in base_path.rglob('*.py'):
        if py_file.name.startswith('_') or '__' in str(py_file):
            continue
        
        rel_path = py_file.relative_to(base_path)
        module_name = '.'.join(rel_path.with_suffix('').parts)
        
        info = analyze_file(py_file)
        if not info:
            continue
            
        modules[module_name] = {
            'file': str(py_file),
            'imports': sorted(list(info['imports'])),
            'classes': info['classes'],
            'functions': info['functions'],
            'async_functions': info['async_functions']
        }
    
    # Generate architecture documentation
    print("=" * 80)
    print("PROMPT-PRIX CURRENT ARCHITECTURE ANALYSIS")
    print("=" * 80)
    
    for module_name, info in sorted(modules.items()):
        print(f"\n### {module_name}")
        print(f"**File:** `{info['file']}`")
        
        if info['imports']:
            print("\n**Imports:**")
            for imp in info['imports']:
                print(f"  - `{imp}`")
        
        if info['classes']:
            print("\n**Classes:**")
            for cls in info['classes']:
                print(f"  - `class {cls}`")
        
        if info['functions'] or info['async_functions']:
            funcs = info['functions'] + info['async_functions']
            print("\n**Functions:**")
            for func in sorted(funcs):
                is_async = func in info['async_functions']
                prefix = "async def" if is_async else "def"
                print(f"  - `{prefix} {func}`")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total modules analyzed: {len(modules)}")
    total_classes = sum(len(info['classes']) for info in modules.values())
    total_functions = sum(len(info['functions'] + info['async_functions']) 
                         for info in modules.values())
    print(f"Total classes: {total_classes}")
    print(f"Total functions: {total_functions}")

if __name__ == '__main__':
    main()
