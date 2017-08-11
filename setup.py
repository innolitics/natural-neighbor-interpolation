from distutils.core import setup, Extension

import numpy.distutils.misc_util


module = Extension(
    'naturalneighbor',
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
    library_dirs=['/usr/local/lib'],
    extra_compile_args=['--std=c++11'],
    sources=['_naturalneighbor.cpp', 'nn.cpp']
)

setup(
    name='NaturalNeighbor',
    version='0.1',
    description='Discrete natural neighbor interpolation in 3D.',
    author='Reece Stevens',
    author_email='rstevens@innolitics.com',
    url='https://github.com/innolitics/natural-neighbor-interpolation',
    long_description='Discrete natural neighbor interpolation in 3D.',
    ext_modules=[module]
)
