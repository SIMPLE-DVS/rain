#!/usr/bin/env python

"""The setup script."""

import versioneer

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "pandas==1.3.0",
    "sklearn==0.0",
    "scikit-learn==0.24.2",
    "pyspark==3.1.2",
    "networkx==2.6.2",
    "matplotlib==3.4.2",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Riccardo Coltrinari",
    author_email="riccardo.coltrinari@studenti.unicam.it",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="SIMPLE Repository is a container for all the nodes used in the SIMPLE Project.",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="simple_repo",
    name="simple_repo",
    packages=find_packages(include=["simple_repo", "simple_repo.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/DazeDC/simple_repo",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
)
