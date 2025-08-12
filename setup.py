from setuptools import setup, find_packages

setup(
    name="data_analysis_utils",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scipy",
        "scikit-learn"
    ],
    author="Your Name",
    description="Reusable Python utilities for quick data exploration and visualization",
    python_requires=">=3.7",
)
