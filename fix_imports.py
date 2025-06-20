import os
import re

# Mapping of old import name to correct module filename (minus .py)
moved = {
    "core.orchestrator":      "core.steg_orchestrator",
    "core.config":            "config.steg_config",
    "core.database":          "core.steg_database",
    "tools.classic_stego":    "tools.classic_stego_tools",
    "tools.image_forensics":  "tools.image_forensics_tools",
    "tools.audio_analysis":   "tools.audio_analysis_tools",
    "tools.file_forensics":   "tools.file_forensics_tools",
    "tools.crypto_analysis":  "tools.crypto_analysis_tools",
    "ai.ml":                  "ai.ml_detector",
    "ai.llm":                 "ai.llm_analyzer",
    "ai.multimodal":          "ai.multimodal_classifier",
    "cloud.integrations":     "cloud.cloud_integrations",
    "utils.gpu":              "utils.gpu_manager",
    # Add more if needed
}

# Build regex for any "from <module> import ..." or "import <module>"
pattern = re.compile(r"from\s+([a-zA-Z0-9_\.]+)\s+import\s+|import\s+([a-zA-Z0-9_\.]+)")

def fix_imports_in_file(fpath):
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(fpath, 'r', encoding='latin1') as f:
            lines = f.readlines()
    orig_lines = list(lines)
    modified = False
    for i, line in enumerate(lines):
        m = pattern.search(line)
        if m:
            modname = m.group(1) or m.group(2)
            if modname in moved:
                fixed = line.replace(modname, moved[modname])
                if fixed != line:
                    lines[i] = fixed
                    modified = True
    if modified:
        print(f"[fix] {fpath}")
        with open(fpath + ".bak", 'w', encoding='utf-8') as f:
            f.writelines(orig_lines)
        with open(fpath, 'w', encoding='utf-8') as f:
            f.writelines(lines)

for root, dirs, files in os.walk('.'):
    for fname in files:
        if fname.endswith('.py'):
            fix_imports_in_file(os.path.join(root, fname))
