#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy


approx_c = Extension('mistofrutta.approx._approx_c',
                    sources = ['mistofrutta/approx/_approx_c.cpp'],
                    include_dirs = ['/home/francesco/.local/lib/boost_1_71_0','/home/francesco/.local/lib/eigen',numpy.get_include()],
                    extra_compile_args=['-ffast-math','-I/opt/boost_1_48_0/include','-Ofast'])

setup(name='mistofrutta',
      version='1.0',
      description='Collection of random utilities',
      author='Francesco Randi',
      author_email='francesco.randi@gmail.com',
      packages=['mistofrutta','mistofrutta.plt','mistofrutta.geometry','mistofrutta.struct','mistofrutta.approx'],
      ext_modules = [approx_c]
     )
