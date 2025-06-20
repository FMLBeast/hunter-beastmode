import os
import re

# Map incorrect import paths to correct ones
REPLACEMENTS = {
    r'from\s+core\.orchestrator\s+import\s+StegOrchestrator':      'from core.steg_orchestrator import StegOrchestrator',
    r'from\s+core\.config\s+import\s+Config':                      'from config.steg_config import Config',
    r'from\s+core\.database\s+import\s+Database':                  'from core.steg_database import Database',
}

# List of imports to comment out if the file doesn't exist
REMOVALS = [
    r'from\s+utils\.logger\s+import\s+.*',
    r'from\s+utils\.system_check\s+import\s+.*',
]

def file_exists_from_import(import_line):
    """
    Checks if an import line (e.g., from core.steg_database import Database) matches a real file.
    """
    m = re.match(r'from\s+([a-zA-Z0-9_\.]+)\s+import\s+.*', import_line)
    if m:
        parts = m.group(1).split('.')
        path = os.path.join(*parts) + '.py'
        return os.path.exists(path)
    return True  # Default to True if not standard pattern

for root, dirs, files in os.walk('.'):
    for fname in files:
        if fname.endswith('.py'):
            fpath = os.path.join(root, fname)
            with open(fpath, 'r') as f:
                lines = f.readlines()

            orig_lines = list(lines)
            modified = False
            for i, line in enumerate(lines):
                # Do replacements for known bad imports
                for bad, good in REPLACEMENTS.items():
                    if re.search(bad, line):
                        lines[i] = re.sub(bad, good, line)
                        modified = True

                # Comment out imports for modules that do not exist
                for bad_pattern in REMOVALS:
                    if re.search(bad_pattern, line):
                        if not file_exists_from_import(line):
                            lines[i] = f"# {line.rstrip()}  # (autofixed: file missing)\n"
                            modified = True

            if modified:
                print(f"[autofix] {fpath}")
                # Backup old file
                bak = fpath + ".bak"
                if not os.path.exists(bak):
                    with open(bak, 'w') as fbak:
                        fbak.writelines(orig_lines)
                # Write new file
                with open(fpath, 'w') as fout:
                    fout.writelines(lines)
