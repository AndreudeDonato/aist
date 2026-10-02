from setuptools import find_packages, setup

setup(
    name="aist",
    version="0.1.0",
    description="Grand-canonical ab initio thermodynamics and motif clustering of gas-phase IryOx clusters",
    author="Andreu A. de Donato",
    url="https://github.com/AndreudeDonato/aist",
    packages=find_packages(include=["aist", "aist.*"]),
    python_requires=">=3.9",
    install_requires=[
        "ase",
        "numpy",
        "scipy",
        "sympy",
        "pymatgen",
        "matplotlib",
        "scikit-learn",
        "umap-learn",
        "dscribe",
    ],
    extras_require={"test": ["pytest"]},
)
