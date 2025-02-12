# -*- coding: utf-8 -*-
# Copyright 2022 Balacoon

from setuptools import setup, find_packages


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="speech_gen_eval",
    version="0.0.1",
    author="Clement Ruhm",
    author_email="clement@balacoon.com",
    description="Collection of tools for evaluation of speech generation",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://balacoon.com/",
    # declare your packages
    packages=find_packages(where="src", exclude=("tests",)),
    package_dir={"": "src"},
    # declare your scripts
    entry_points="""\
     [console_scripts]
     speech-gen-eval = speech_gen_eval.main:main
    """
)
