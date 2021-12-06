# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ThymeBoost",
    version="0.1.6",
    author="Tyler Blume",
    url="https://github.com/tblume1992/ThymeBoost",
    long_description=long_description,
    long_description_content_type="text/markdown",
    description = "Spicy time series forecasting.",
    author_email = 'tblume@mail.USF.edu', 
    keywords = ['forecasting', 'time series', 'seasonality', 'trend'],
      install_requires=[           
                        'numpy',
                        'pandas',
                        'statsmodels',
                        'scikit-learn',
                        'scipy',
                        'more-itertools',
                        'matplotlib',
                        'tqdm'
                        ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)


