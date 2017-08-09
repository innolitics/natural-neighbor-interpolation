from distutils.core import setup, Extension
import numpy.distutils.misc_util

module = Extension('naturalneighbor',
                    # include_dirs = ['/usr/local/include'],
                    include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs(),
                    # libraries = ['boost'],
                    library_dirs = ['/usr/local/lib', '/usr/local/Cellar/boost/1.64.0_1/lib/'],
                    extra_compile_args = ['--std=c++11'],
                    sources = ['_naturalneighbor.cpp', 'nn.cpp'])

setup (name = 'NaturalNeighbor',
       version = '1.0',
       description = 'A module for performing natural neighbor interpolation.',
       author = 'Reece Stevens',
       author_email = 'rstevens@innolitics.com',
       url = 'https://github.com/innolitics/natural-neighbor-interpolation',
       long_description = '''
A module for performing natural neighbor interpolation.
''',
       ext_modules = [module])
