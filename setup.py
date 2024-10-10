"""This module sets up the package for distribution."""
from setuptools import setup, find_packages

setup(
    name='ricci_flow_reduction',
    version='0.1',
    packages=find_packages(),
    description='A library for dimensionality reduction using Ricci Flow',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Giorgio Micaletto',
    author_email='giorgio.micaletto@studbocconi.it',
    url='https://github.com/GiorgioMB/RicciFlowDimReduction/',
    install_requires=[
        'numpy>=1.23',
        'GraphRicciCurvature>=0.5.3.2',
        'scikit-learn>=1.5.1',
        'networkx>=3.3'
    ],
    python_requires='>=3.6',
)
