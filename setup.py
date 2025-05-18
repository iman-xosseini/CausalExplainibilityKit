# setup.py
from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    required = f.read().splitlines()

# Try to get version from __init__.py
version = {}
with open("causal_explainer_kit/__init__.py") as fp:
    exec(fp.read(), version)


setup(
    name="CausalExplainabilityKit",
    version=version["__version__"],  # Get version from __init__.py
    author="Iman Hosseini",  # Change this
    author_email="iman.hosseini@ijs.si",  # Change this
    description="A Python toolkit for causal explanation of machine learning models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/iman-xosseini/CausalExplainibilityKit",  # Change this
    packages=find_packages(
        exclude=["examples*", "tests*"]
    ),  # find_packages will find 'causal_explainer_kit'
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",  # Choose your license
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.11",  # Specify your minimum Python version
)
