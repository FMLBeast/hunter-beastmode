from setuptools import setup, find_packages
setup(
    name="hunter-steg",
    version="1.0",
    description="Enterprise-Grade Steganography & Forensics Platform",
    author="Hunter Project",
    packages=find_packages(),
    install_requires=[l.strip() for l in open("requirements.txt")],
    include_package_data=True,
    python_requires='>=3.10',
)
