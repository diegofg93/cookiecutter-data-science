#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os
from pathlib import Path

from setuptools import find_packages, setup

# Package meta-data.
NAME = 'src'
REQUIRES_PYTHON = '>=3.6.0'


# What packages are required for this module to be executed?
def list_reqs(fname='requirements.txt'):
    with open(fname) as fd:
        return fd.read().splitlines()


# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the
# Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = '{{ cookiecutter.description }}'


# Load the package's __version__.py module as a dictionary.
ROOT_DIR = Path(__file__).parent.resolve()
PACKAGE_DIR = ROOT_DIR / NAME
about = {}
with open(ROOT_DIR / 'VERSION') as f:
    _version = f.read().strip()
    about['__version__'] = _version


setup(
    name='src',
    description='{{ cookiecutter.description }}',
    author='{{ cookiecutter.author_name }}',
    author_email= "...",
    version=about['__version__'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='{% if cookiecutter.open_source_license == 'MIT' %}MIT{% elif cookiecutter.open_source_license == 'BSD-3-Clause' %}BSD-3{% endif %}',
    python_requires=REQUIRES_PYTHON,
    url='...',
    packages=find_packages(exclude=('tests',)),
    package_data={'src': ['VERSION'],
                  'config': ['*']},
    install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
