#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy


approx_c = Extension('mistofrutta.approx._approx_c',
                    sources = ['mistofrutta/approx/_approx_c.cpp'],
                    include_dirs = [numpy.get_include()],
                    extra_compile_args=['-ffast-math','-Ofast'])
                    
_ft = Extension('mistofrutta.ft._ft',
                    sources = ['mistofrutta/ft/_ft.cpp','mistofrutta/ft/ft.cpp'],
                    include_dirs = [numpy.get_include()],
                    extra_compile_args=['-ffast-math','-Ofast'])

setup(name='mistofrutta',
      version='1.0',
      description='Collection of random utilities',
      author='Francesco Randi',
      author_email='francesco.randi@gmail.com',
      packages=['mistofrutta','mistofrutta.plt','mistofrutta.geometry','mistofrutta.struct','mistofrutta.approx','mistofrutta.ft'],
      ext_modules = [approx_c, _ft]
     )
