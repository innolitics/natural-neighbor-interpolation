from distutils.core import setup, Extension

module = Extension('natural_neighbor',
                    include_dirs = ['/usr/local/include'],
                    # libraries = ['boost'],
                    library_dirs = ['/usr/local/lib', '/usr/local/Cellar/boost/1.64.0_1/lib/'],
                    extra_compile_args = ['--std=c++11'],
                    sources = ['nn.cpp'])

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
