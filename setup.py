"""Pip installation script for `gemo`."""

import os
import re
from setuptools import find_packages, setup

package_name = 'gemo'


def get_version():

    ver_file = '{}/_version.py'.format(package_name)
    with open(ver_file) as handle:
        ver_str_line = handle.read()

    ver_pattern = r'^__version__ = [\'"]([^\'"]*)[\'"]'
    match = re.search(ver_pattern, ver_str_line, re.M)
    if match:
        ver_str = match.group(1)
    else:
        msg = 'Unable to find version string in "{}"'.format(ver_file)
        raise RuntimeError(msg)

    return ver_str


def get_long_description():

    readme_file = 'README.md'
    with open(readme_file, encoding='utf-8') as handle:
        contents = handle.read()

    return contents


setup(
    name=package_name,
    version=get_version(),
    description=('Visualise sets of geometric objects.'),
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    author='Adam J. Plowman',
    author_email='adam.plowman@manchester.ac.uk',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'plotly',           # for backend='plotly'
        'notebook>=5.3',    # for plotly in Jupyter NBs/Lab
        'ipywidgets>=7.5',  # for plotly in Jupyter NBs/Lab
        'jupyterlab>=1.0',  # for plotly in Jupyter NBs/Lab
        'vecmaths',
        'spatial_sites>=0.2.0',
    ],
    project_urls={
        'Github': 'https://github.com/aplowman/gemo',
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Operating System :: OS Independent',
    ],
)
