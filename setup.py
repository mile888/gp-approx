from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

DESCRIPTION = 'Gaussian Process regression with analytical approximation of uncertainty propagation.'

# Setting up
setup(
    name="gp-approx",
    version='1.0',
    author="Mile Mitrovic",
    author_email="mitrovich888@gmail.com",
    description=DESCRIPTION,
    long_description = long_description,
    url = 'https://github.com/mile888/gp-approx',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'pandas'],
    keywords=['machine learning', 'gaussian process', 'python', 'taylor approximation', 'moment matching'],
    classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved",
            "Programming Language :: Python",
            "License :: OSI Approved ",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            "Development Status :: 5 - Production/Stable",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: Unix",
            "Operating System :: MacOS",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: Implementation :: PyPy",
                ]
)