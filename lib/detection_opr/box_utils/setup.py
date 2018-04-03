#!/usr/bin/env python3
# coding=utf-8
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

package = Extension('bbox', ['bbox.pyx'])
setup(ext_modules=cythonize([package]), include_dirs=[np.get_include()])
