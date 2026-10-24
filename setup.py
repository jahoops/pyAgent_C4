from setuptools import setup, Extension
import sysconfig

# Get the include and library directories for Python
include_dirs = [sysconfig.get_path('include')]
library_dirs = [sysconfig.get_config_var('LIBDIR')]

module = Extension(
    'addtwo',
    sources=['addtwo.c'],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=['python' + sysconfig.get_python_version()]
)

setup(
    name='addtwo',
    version='1.0',
    description='A Python module that adds two numbers',
    ext_modules=[module],
)