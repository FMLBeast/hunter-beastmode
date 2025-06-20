# Hunter StegAnalyzer

**Enterprise-Grade, GPU-Accelerated Steganography, Forensics & AI Pipeline**

## What is Hunter StegAnalyzer?

Hunter is a next-gen steganalysis and digital forensics platform built for massive parallelism, enterprise-scale datasets, CTF challenges, anti-fraud, and malware research.

It fuses:
- Classic CLI tools (steghide, outguess, binwalk, zsteg, stegseek, etc.)
- Deep learning & GPU (Noiseprint, custom CNN, Vision Transformers, audio ML)
- LLM/AI (OpenAI/Anthropic/HF/Local) for reasoning and triage
- File carving, crypto analysis, advanced entropy, and pattern detection
- Cloud integrations (VirusTotal, NSRL, Hashlookup, etc.)
- Live web dashboard, auto-reporting, and resumable pipelines

## Installation (Vast.ai, Linux, or baremetal)

```bash
git clone <your-git-url> hunter-steg
cd hunter-steg
bash install.sh
```

## Quick Start

```bash
# After install (Python 3.10+, Ubuntu 22.04+, GPU highly recommended)
source .venv/bin/activate
python steg_main.py --file suspicious.png
python steg_main.py --dir /mnt/massive-corpus/
```

Dashboard: http://127.0.0.1:8080
