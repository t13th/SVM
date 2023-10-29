cmake_minimum_required(VERSION 3.22)

project(SVM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
string(TIMESTAMP COMPILE_TIME %Y%m%d-%H%M%S)

add_definitions(-D__DEBUG__)

if(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))
  add_definitions(-D__DEBUG__)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  add_compile_options(/utf-8)
endif()

add_compile_options(-static)

include_directories(modules/*)
file(GLOB MODULES modules/*/*.cpp)

file(GLOB DEMO demo/*/*.cpp)
foreach(FILE ${DEMO})
    # 使用文件名作为可执行文件的名称
    get_filename_component(FILE_DIR ${FILE} DIRECTORY)
    get_filename_component(EXECUTABLE_NAME ${FILE_DIR} NAME)
    add_executable(${EXECUTABLE_NAME} ${FILE} ${MODULES})
endforeach()