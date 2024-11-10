cmake_minimum_required(VERSION 3.18)

project(MarchingCubes LANGUAGES CXX CUDA)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

list(PREPEND CMAKE_MODULE_PATH
        ${CMAKE_CURRENT_LIST_DIR}/cmake)
include(eigen)

add_subdirectory(src)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE SVO)
