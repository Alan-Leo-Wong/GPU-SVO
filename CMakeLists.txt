cmake_minimum_required(VERSION 3.18)

project(MarchingCubes LANGUAGES CXX CUDA)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(PREPEND CMAKE_MODULE_PATH
        ${CMAKE_CURRENT_LIST_DIR}/cmake)
include(eigen)

add_library(BaseModel STATIC BaseModel.h BaseModel.cpp)
target_link_libraries(BaseModel PUBLIC Eigen3::Eigen)
if (MSVC)
    target_compile_options(BaseModel PRIVATE /wd4068)
endif()

add_library(SVO STATIC CUDAUtil.h SVO.h SVO.cu)
target_include_directories(SVO PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(SVO PRIVATE BaseModel)
target_compile_features(SVO PRIVATE cuda_std_17)
target_compile_options(SVO PRIVATE -diag-suppress=177 -diag-suppress=20012 -diag-suppress=20015 -diag-suppress=20013)
if(MSVC)
    target_compile_options(SVO PRIVATE -Xcompiler /wd4068)
endif ()

add_executable(main main.cpp)
target_link_libraries(main PRIVATE SVO)
