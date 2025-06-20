import os
import sys
import ast
import importlib.util

# Ignore these folders (add more if needed)
IGNORE_DIRS = {'.venv', 'env', 'venv', '__pycache__', '.git'}

def is_local_module(module_name, base_path):
    """
    Checks if a module can be found as a file in the project, relative to base_path.
    E.g., core.steg_orchestrator => core/steg_orchestrator.py
    """
    parts = module_name.split('.')
    if not parts:
        return False
    # Try absolute from project root
    project_root = os.path.abspath(base_path)
    rel_path = os.path.join(project_root, *parts) + '.py'
    return os.path.isfile(rel_path)

def check_imports_in_file(fpath, project_root):
    errors = []
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            src = f.read()
    except Exception as e:
        errors.append(f"{fpath}: [READ ERROR] {e}")
        return errors
    try:
        tree = ast.parse(src, fpath)
    except Exception as e:
        errors.append(f"{fpath}: [PARSE ERROR] {e}")
        return errors

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modname = alias.name
                if is_local_module(modname, project_root):
                    continue  # local file exists
                # Try to import (system or pip)
                if importlib.util.find_spec(modname) is None:
                    errors.append(f"{fpath}: MISSING MODULE '{modname}'")
        elif isinstance(node, ast.ImportFrom):
            modname = node.module
            if modname is None:  # 'from . import foo'
                continue
            if is_local_module(modname, project_root):
                continue  # local file exists
            if importlib.util.find_spec(modname) is None:
                errors.append(f"{fpath}: MISSING MODULE '{modname}'")
    return errors

def main():
    project_root = os.path.abspath('.')
    for root, dirs, files in os.walk(project_root):
        # Ignore venv and other skip-dirs
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for fname in files:
            if fname.endswith('.py'):
                fpath = os.path.join(root, fname)
                errors = check_imports_in_file(fpath, project_root)
                for err in errors:
                    print(err)

if __name__ == "__main__":
    main()
