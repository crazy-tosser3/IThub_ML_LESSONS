cmake_minimum_required(VERSION 4.3)
project(RectangleFound)

set(CMAKE_CXX_STANDARD 26)

find_package(OpenCV REQUIRED)

add_executable(RectangleFound main.cpp)

target_link_libraries(RectangleFound PRIVATE ${OpenCV_LIBS})
