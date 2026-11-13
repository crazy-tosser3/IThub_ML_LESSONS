cmake_minimum_required(VERSION 3.18)
project(Razukrashka2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Поиск OpenCV
find_package(OpenCV REQUIRED)

# 2. Настройка локального ONNX Runtime из third_party/onnxruntime
set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime")
set(ORT_INCLUDE_DIR "${ORT_ROOT}/include")
set(ORT_LIB_DIR "${ORT_ROOT}/lib")

if(APPLE)
    set(ORT_LIBRARY "${ORT_LIB_DIR}/libonnxruntime.dylib")
elseif(WIN32)
    set(ORT_LIBRARY "${ORT_LIB_DIR}/onnxruntime.lib")
else()
    set(ORT_LIBRARY "${ORT_LIB_DIR}/libonnxruntime.so")
endif()

add_executable(Razukrashka2 main.cpp)

target_include_directories(Razukrashka2 PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${ORT_INCLUDE_DIR}
)

target_link_libraries(Razukrashka2 PRIVATE
        ${OpenCV_LIBS}
        ${ORT_LIBRARY}
)

# Прописываем RPATH, чтобы бинарник из cmake-build-debug сразу находил libonnxruntime.1.31.1.dylib
set_target_properties(Razukrashka2 PROPERTIES
        BUILD_RPATH "${ORT_LIB_DIR}"
        INSTALL_RPATH "${ORT_LIB_DIR}"
)