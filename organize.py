import os
import shutil

file_moves = {
    "steg_main.py":                 "steg_main.py",
    "steg_config.py":               "config/steg_config.py",
    "steg_database.py":             "core/steg_database.py",
    "dashboard.py":                 "core/dashboard.py",
    "reporter.py":                  "core/reporter.py",
    "checkpoint_manager.py":        "utils/checkpoint_manager.py",
    "file_analyzer.py":             "core/file_analyzer.py",
    "graph_tracker.py":             "core/graph_tracker.py",
    "classic_stego_tools.py":       "tools/classic_stego_tools.py",
    "image_forensics_tools.py":     "tools/image_forensics_tools.py",
    "audio_analysis_tools.py":      "tools/audio_analysis_tools.py",
    "file_forensics_tools.py":      "tools/file_forensics_tools.py",
    "crypto_analysis_tools.py":     "tools/crypto_analysis_tools.py",
    "ml_detector.py":               "ai/ml_detector.py",
    "llm_analyzer.py":              "ai/llm_analyzer.py",
    "multimodal_classifier.py":     "ai/multimodal_classifier.py",
    "steg_orchestrator.py":         "core/steg_orchestrator.py",
    "cloud_integrations.py":        "cloud/cloud_integrations.py",
    "gpu_manager.py":               "utils/gpu_manager.py",
}

folders_needed = set(os.path.dirname(dst) for dst in file_moves.values() if os.path.dirname(dst))
for d in ["logs", "reports", "data", "models", "wordlists", "static", "templates"]:
    folders_needed.add(d)

for folder in folders_needed:
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

for old, new in file_moves.items():
    if os.path.exists(old):
        target = os.path.join(new)
        if os.path.exists(target):
            print(f"[!] {target} already exists, skipping.")
            continue
        shutil.move(old, target)
        print(f"[+] Moved {old} -> {target}")

requirements = (
    "torch\n"
    "torchvision\n"
    "torchaudio\n"
    "numpy\n"
    "opencv-python\n"
    "Pillow\n"
    "matplotlib\n"
    "scikit-image\n"
    "python-magic\n"
    "librosa\n"
    "soundfile\n"
    "aiohttp\n"
    "fastapi\n"
    "uvicorn\n"
    "sqlalchemy\n"
    "asyncpg\n"
    "neo4j\n"
    "ssdeep\n"
    "yara-python\n"
    "pycryptodome\n"
    "scikit-learn\n"
    "tesseract\n"
    "pytesseract\n"
    "transformers\n"
    "timm\n"
    "networkx\n"
)
with open("requirements.txt", "w") as f:
    f.write(requirements)

gitignore = (
    "# Python\n"
    "__pycache__/\n"
    "*.py[cod]\n"
    "*.so\n"
    ".venv/\n"
    "env/\n"
    "*.egg-info/\n"
    ".eggs/\n"
    ".Python\n"
    ".ipynb_checkpoints/\n"
    "# IDE\n"
    ".vscode/\n"
    ".idea/\n"
    "*.sublime-project\n"
    "*.sublime-workspace\n"
    "# OS\n"
    ".DS_Store\n"
    "Thumbs.db\n"
    "# Logs & Output\n"
    "logs/\n"
    "reports/\n"
    "*.log\n"
    "*.sqlite\n"
    "*.db\n"
    "data/\n"
    "models/\n"
    "wordlists/\n"
    "output/\n"
    "*.tmp\n"
    "# Secrets\n"
    "*.key\n"
    "*.pem\n"
    ".env\n"
    "# Jupyter\n"
    "*.ipynb\n"
    "# Docker\n"
    "docker-compose.override.yml\n"
)
with open(".gitignore", "w") as f:
    f.write(gitignore)

license_text = (
    "MIT License\n\n"
    "Copyright (c) 2025 Hunter\n\n"
    "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
    "of this software and associated documentation files (the \"Software\"), to deal\n"
    "in the Software without restriction, including without limitation the rights\n"
    "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
    "copies of the Software, and to permit persons to whom the Software is\n"
    "furnished to do so, subject to the following conditions:\n\n"
    "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n"
    "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
    "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.\n"
)
with open("LICENSE", "w") as f:
    f.write(license_text)

readme = (
    "# Hunter StegAnalyzer\n\n"
    "**Enterprise-Grade, GPU-Accelerated Steganography, Forensics & AI Pipeline**\n\n"
    "## What is Hunter StegAnalyzer?\n\n"
    "Hunter is a next-gen steganalysis and digital forensics platform built for massive parallelism, enterprise-scale datasets, CTF challenges, anti-fraud, and malware research.\n\n"
    "It fuses:\n"
    "- Classic CLI tools (steghide, outguess, binwalk, zsteg, stegseek, etc.)\n"
    "- Deep learning & GPU (Noiseprint, custom CNN, Vision Transformers, audio ML)\n"
    "- LLM/AI (OpenAI/Anthropic/HF/Local) for reasoning and triage\n"
    "- File carving, crypto analysis, advanced entropy, and pattern detection\n"
    "- Cloud integrations (VirusTotal, NSRL, Hashlookup, etc.)\n"
    "- Live web dashboard, auto-reporting, and resumable pipelines\n\n"
    "## Installation (Vast.ai, Linux, or baremetal)\n\n"
    "```bash\n"
    "git clone <your-git-url> hunter-steg\n"
    "cd hunter-steg\n"
    "bash install.sh\n"
    "```\n\n"
    "## Quick Start\n\n"
    "```bash\n"
    "# After install (Python 3.10+, Ubuntu 22.04+, GPU highly recommended)\n"
    "source .venv/bin/activate\n"
    "python steg_main.py --file suspicious.png\n"
    "python steg_main.py --dir /mnt/massive-corpus/\n"
    "```\n\n"
    "Dashboard: http://127.0.0.1:8080\n"
)
with open("README.md", "w") as f:
    f.write(readme)

install_sh = (
    "#!/bin/bash\n"
    "set -e\n\n"
    "echo \"==[ Hunter StegAnalyzer Install Script ]==\"\n"
    "sudo apt update && sudo apt install -y \\\n"
    "  python3-pip python3-venv python3-dev build-essential \\\n"
    "  libmagic-dev libopencv-dev exiftool binwalk foremost \\\n"
    "  steghide outguess ffmpeg sox tesseract-ocr zsteg \\\n"
    "  git curl unzip libsqlite3-dev graphviz imagemagick \\\n"
    "  libsm6 libxext6 libxrender-dev\n\n"
    "python3 -m venv .venv\n"
    "source .venv/bin/activate\n"
    "pip install --upgrade pip setuptools wheel\n"
    "pip install -r requirements.txt\n\n"
    "for d in models wordlists logs reports data static templates; do\n"
    "    [ -d \"$d\" ] || mkdir \"$d\"\n"
    "done\n\n"
    "echo \"[*] Installation complete.\"\n"
    "echo \"To run a scan:\"\n"
    "echo \"  source .venv/bin/activate\"\n"
    "echo \"  python steg_main.py --file suspicious.png\"\n"
    "echo \"  python steg_main.py --dir /data/batch/\"\n"
    "echo \"Dashboard: http://127.0.0.1:8080\"\n"
)
with open("install.sh", "w") as f:
    f.write(install_sh)
os.chmod("install.sh", 0o755)

setup_py = (
    "from setuptools import setup, find_packages\n"
    "setup(\n"
    "    name=\"hunter-steg\",\n"
    "    version=\"1.0\",\n"
    "    description=\"Enterprise-Grade Steganography & Forensics Platform\",\n"
    "    author=\"Hunter Project\",\n"
    "    packages=find_packages(),\n"
    "    install_requires=[l.strip() for l in open(\"requirements.txt\")],\n"
    "    include_package_data=True,\n"
    "    python_requires='>=3.10',\n"
    ")\n"
)
with open("setup.py", "w") as f:
    f.write(setup_py)

print("\n[✔] Pro structure created, all files moved, all pro-level metadata added.\nReady for git init, add, commit, and push.")
