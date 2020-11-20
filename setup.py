import codecs  # To use a consistent encoding
import os
from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with codecs.open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    install_requires = []
else:
    with open('requirements.txt') as f:
        install_requires = [l.strip() for l in f]

setup(
    name="diagNNose",
    version="1.0",
    description="A library that facilitates a broad set of tools for analysing "
                "hidden activations of neural models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.6",
    ],
    keywords="interpretability, analysis, deep learning, nlp",
    url="https://github.com/i-machine-think/diagnnose",
    packages=find_packages(exclude=["contrib", "docs", "test", "scripts"]),
    install_requires=install_requires,
    python_requires=">=3.6.0",
    extras_require={"dev": ["check-manifest"], "test": ["coverage"]},
)
