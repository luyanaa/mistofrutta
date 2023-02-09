#!/usr/bin/env python

from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext
import os

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
					

if os.name == "nt":
	os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = 'C:\Program Files\Git\cmd\git.exe'

	import numpy
	approx_c.include_dirs.append(numpy.get_include())
	_ft.include_dirs.append(numpy.get_include())
	_convolve.include_dirs.append(numpy.get_include())

import git
# Get git commit info to build version number/tag
repo = git.Repo('.git')
git_hash = repo.head.object.hexsha
git_url = repo.remotes.origin.url
v = repo.git.describe(always=True)
if repo.is_dirty(): v += ".dirty"

requirements = [
    "numpy",
    "matplotlib",
    "shapely",
]


setup(name='mistofrutta',
      version=v,
      description='Collection of random utilities',
      author='Francesco Randi',
      author_email='francesco.randi@gmail.com',
      packages=['mistofrutta','mistofrutta.plt','mistofrutta.geometry','mistofrutta.struct','mistofrutta.approx','mistofrutta.ft', 'mistofrutta.num'],
      install_requires=requirements,
      ext_modules = [approx_c, _ft, _convolve]
     )
