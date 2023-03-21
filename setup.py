import setuptools_scm  # noqa: F401
from setuptools import setup, find_packages


setup(
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
