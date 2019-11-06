#!/usr/bin/env python

from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext
import git

class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""

    def run(self):

        # Import numpy here, only when headers are needed
        import numpy
        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())
        
        # Call original build_ext command
        build_ext.run(self)


approx_c = Extension('mistofrutta.approx._approx_c',
                    sources = ['mistofrutta/approx/_approx_c.cpp'],
                    include_dirs = [],
                    extra_compile_args=['-ffast-math','-Ofast'])
                    
_ft = Extension('mistofrutta.ft._ft',
                    sources = ['mistofrutta/ft/_ft.cpp','mistofrutta/ft/ft.cpp'],
                    include_dirs = [],
                    extra_compile_args=['-ffast-math','-Ofast'])
                    
_convolve = Extension('mistofrutta.convolve._convolve',
                    sources = ['mistofrutta/convolve/_convolve.cpp','mistofrutta/convolve/convolve.cpp'],
                    include_dirs = [],
                    extra_compile_args=['-ffast-math','-Ofast'])

setup(name='mistofrutta',
      version='1.0',
      description='Collection of random utilities',
      author='Francesco Randi',
      author_email='francesco.randi@gmail.com',
      packages=['mistofrutta','mistofrutta.plt','mistofrutta.geometry','mistofrutta.struct','mistofrutta.approx','mistofrutta.ft'],
      ext_modules = [approx_c, _ft, _convolve]
     )
