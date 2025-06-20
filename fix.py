#!/usr/bin/env python3
"""
Fix script to replace config.performance with config.orchestrator
"""

import os
import re
import json

# Define replacements for .performance. to .orchestrator. ONLY for concurrency-related settings
REPLACEMENTS = [
    (r'(\bconfig|self\.config)\.performance\.max_concurrent_files', r'\1.orchestrator.max_concurrent_files'),
    (r'(\bconfig|self\.config)\.performance\.max_concurrent_tasks', r'\1.orchestrator.max_cpu_workers'),
    (r'(\bconfig|self\.config)\.performance\.', r'\1.orchestrator.'),  # Catch-all for other performance refs
    (r"(\bconfig|self\.config)\['classic_stego'\]", r'\1.classic_stego'),
    (r"(\bconfig|self\.config)\['file_forensics'\]", r'\1.file_forensics'),
    (r"(\bconfig|self\.config)\['orchestrator'\]", r'\1.orchestrator'),
    (r"(\bconfig|self\.config)\['database'\]", r'\1.database'),
]

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        original = f.read()
    fixed = original

    # Replace dict-style config access with attribute access
    for pattern, repl in REPLACEMENTS:
        fixed = re.sub(pattern, repl, fixed)

    # Fix resume logic: ensure dict to Config update uses JSON decode if needed
    fixed = re.sub(
        r'self\.config\.update\((session\["config"\])\)',
        'config_data = \\1\nif isinstance(config_data, str):\n    config_data = json.loads(config_data)\nself.config.update(config_data)',
        fixed,
    )

    # Only write back if changed
    if fixed != original:
        print(f'[fix] {filepath}')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed)

def walk_and_fix(base='.'):
    for root, dirs, files in os.walk(base):
        # Skip virtualenvs and git
        if '.venv' in dirs: dirs.remove('.venv')
        if '.git' in dirs: dirs.remove('.git')
        for name in files:
            if name.endswith('.py'):
                fix_file(os.path.join(root, name))

if __name__ == '__main__':
    print('Auto-fixing config usage in repo...')
    walk_and_fix()
    print('Done.')
