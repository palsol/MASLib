from distutils.core import setup, Extension
import numpy

setup(name='mas', version='1.0', \
      ext_modules=[Extension('mas',
                             sources=['mainPython.cpp', 'cpp/masCone.cpp'],
                             extra_compile_args=['-std=c++11'],
                             include_dirs=['/usr/local/include',
                                           '/home/palsol/anaconda3/lib/python3.6/site-packages/numpy/core/include',
                                           numpy.get_include()])])