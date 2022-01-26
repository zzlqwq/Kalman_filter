cmake_minimum_required(VERSION 3.19)
project(kalman_filter)

set(CMAKE_CXX_STANDARD 14)

include_directories("eigen-3.4.0")

add_executable(kalman_filter kalman.h kalman.cpp main.cpp)