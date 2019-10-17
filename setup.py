# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io

import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import setup
from setuptools import find_packages


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")
    ) as fh:
        return fh.read()


setup(
    name="mt-dnn",
    license="MIT",
    description="Multi-Task Deep Neural Networks for Natural Language Understanding",
    long_description="%s"
    % (
        re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub(
            "", read("README.md")
        )
    ),
    author="Xiaodong Liu",
    author_email="xiaodl@microsoft.com",
    url="https://github.com/namisan/mt-dnn",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Utilities",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Telecommunications Industry",
    ],
    project_urls={
        "Documentation": "https://github.com/namisan/mt-dnn/",
        "Issue Tracker": "https://github.com/namisan/mt-dnn/issues",
    },
    keywords=[
        "Microsoft NLP",
        "Natural Language Processing",
        "Text Processing",
        "Word Embedding",
        "Multi-Task Deep Neural Networks",
        "Microsoft Research",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "torch==1.1.0",
        "tqdm",
        "colorlog",
        "boto3",
        "pytorch-pretrained-bert==v0.6.0",
        "regex",
        "scikit-learn",
        "pyyaml",
        "pytest",
        "sentencepiece",
        "tensorboardX",
        "tensorboard",
        "future",
        "apex",
        "fairseq",
    ],
    entry_points={"console_scripts": ["mt-dnn = mt_dnn.cli.main"]},
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
)

