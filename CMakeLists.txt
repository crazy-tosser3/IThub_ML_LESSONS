cmake_minimum_required(VERSION 4.3)
project(FaceRecognition)

set(CMAKE_CXX_STANDARD 26)

find_package(OpenCV REQUIRED)

add_executable(FaceRecognition main.cpp)

target_link_libraries(FaceRecognition PRIVATE ${OpenCV_LIBS})