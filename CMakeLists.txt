cmake_minimum_required(VERSION 3.6)
project(MASLib)

set(CMAKE_CXX_STANDARD 11)
include_directories(contrib)
set(SOURCE_FILES include/masCone.h cpp/masCone.cpp main.cpp)
add_executable(MASLib ${SOURCE_FILES})