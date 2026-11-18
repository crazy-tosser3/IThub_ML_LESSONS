cmake_minimum_required(VERSION 4.3)
project(final)

set(CMAKE_CXX_STANDARD 26)

find_package(OpenCV REQUIRED)

add_executable(final main.cpp)

target_link_libraries(final PRIVATE ${OpenCV_LIBS})