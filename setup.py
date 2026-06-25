from setuptools import setup, find_packages

setup(
    name="heatmap-line-extraction",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)