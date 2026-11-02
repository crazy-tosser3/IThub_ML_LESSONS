cmake_minimum_required(VERSION 4.3)
project(TaskOne)

set(CMAKE_CXX_STANDARD 26)

find_package(OpenCV REQUIRED)

add_executable(TaskOne main.cpp)

target_link_libraries(TaskOne PRIVATE ${OpenCV_LIBS})