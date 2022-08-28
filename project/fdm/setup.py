#!/usr/bin/env python3

import pkg_resources
pkg_resources.require(['pip >= 22'])

from setuptools import setup

setup(name='fdm',
      author="Maikel D.F.",
      author_email="maikeldf@gmail.com",
      version='0.0.1',
      py_modules=["fdm"],
      python_requires=">=3.9",
      install_requires=['torch',
                        'stable-baselines3',
                        'gym',
                        'tensorflow',
                        'tensorboard']
                  )