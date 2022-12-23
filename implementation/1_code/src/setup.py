from setuptools import setup, find_packages


"""
To make src a package, so that in Jupyter Notebooks, modules in this package
can be directly accessed relatively from src
"""


setup(
    name="src",
    version="0.0.1",
    packages=find_packages(),
)
