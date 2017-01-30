from distutils.core import setup, Extension

setup(name='mas', version='1.0', \
      ext_modules=[Extension('mas',
                             sources=['main.cpp', 'cpp/masCone.cpp'],
                             extra_compile_args=['-std=c++11'],
                             include_dirs=['/usr/local/include', '/usr/include/numpy'])])
