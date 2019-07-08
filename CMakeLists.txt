cmake_minimum_required(VERSION 3.14)
project(nonlocal_assembly C)

set(CMAKE_C_STANDARD 11)

include_directories(.)

add_executable(nonlocal_assembly
    test.c)
