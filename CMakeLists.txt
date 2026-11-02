cmake_minimum_required(VERSION 4.3)
project(TaskTwo)

set(CMAKE_CXX_STANDARD 26)

find_package(OpenCV REQUIRED)

add_executable(TaskTwo main.cpp)

target_link_libraries(TaskTwo PRIVATE ${OpenCV_LIBS})