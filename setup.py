from setuptools import setup, find_packages

setup(
    name="aist",                          # Your package name
    version="0.1.0",                     # Starting version
    packages=find_packages(),            # Automatically find packages in your directory
    install_requires=[
        "ase",
        "numpy",
        "matplotlib",
        "sympy",
        "scipy",
        "pymatgen",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A short description of your aist package",
    url="https://github.com/yourusername/aist",  # Optional, add your repo URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Change license if needed
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
