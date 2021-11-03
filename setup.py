#!/usr/bin/env python
from setuptools import setup

setup(
    name='EFGM',
    version='0.1.0',
    author='Mohammed Parvez',
    author_email='parvezmushtaak@gmail.com',
    description='A Package to implement Element Free Galerkin Method',
    long_description=open('README.md').read(),
    install_requires=["numpy", "pytest"],

)
