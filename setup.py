#!/usr/bin/env python

"""
 Copyright (C) 2023 Università degli Studi di Camerino and Sigma S.p.A.
 Authors: Alessandro Antinori, Rosario Capparuccia, Riccardo Coltrinari, Flavio Corradini, Marco Piangerelli, Barbara Re, Marco Scarpetta

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Affero General Public License as
 published by the Free Software Foundation, either version 3 of the
 License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
 """

"""The setup script."""

import versioneer
from modules_requirements import get_modules_requirements

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("requirements-dev.txt") as f:
    requirements_dev = f.read().splitlines()

with open("requirements-docs.txt") as f:
    requirements_docs = f.read().splitlines()

with open("requirements-test.txt") as f:
    requirements_test = f.read().splitlines()

modules = get_modules_requirements()

extras_require = {
    "dev": requirements_dev,
    "docs": requirements_docs,
    "test": requirements_test,
    "full": requirements_dev + requirements_docs + requirements_test,
}

for module in modules:
    extras_require[module] = modules[module]
    extras_require["full"].extend(modules[module])

setup(
    author="Università degli Studi di Camerino and Sigma S.p.A.",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "LICENSE :: OSI APPROVED :: GNU AFFERO GENERAL PUBLIC LICENSE V3 OR LATER (AGPLV3+)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="SIMPLE Repository is a container for all the nodes used in the SIMPLE Project.",
    install_requires=requirements,
    extras_require=extras_require,
    license="GNU Affero General Public License v3 or later (AGPLv3+)",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="rain",
    name="rain",
    packages=find_packages(include=["rain", "rain.*"]),
    test_suite="tests",
    tests_require=requirements_test,
    url="https://github.com/SIMPLE-DVS/rain",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
)
