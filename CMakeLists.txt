cmake_minimum_required(VERSION 3.22)

project(SVM)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
string(TIMESTAMP COMPILE_TIME %Y%m%d-%H%M%S)

find_package(Eigen3 3.4 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})
# find_package(BLAS REQUIRED)
# find_package(MKL REQUIRED)
# include_directories(${MKL_INCLUDE})
# link_directories(${MKL_LIBRARIES})

# add_definitions(-D__USE_EIGEN__)
# add_definitions(-DEIGEN_USE_MKL_ALL)
# add_definitions(-DEIGEN_USE_BLAS)
# add_definitions(-DEIGEN_USE_GPU)
# link_libraries(${mkl_core_file} ${mkl_intel_ilp64_file} "-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl")

add_definitions(-DPROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/")

if(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fsanitize=address")
  add_definitions(-D__DEBUG__)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Ofast -march=native")
  add_definitions(-DNDEBUG)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  add_compile_options(/utf-8)
endif()

include_directories(modules/* modules)
file(GLOB MODULES modules/*/*.cpp)


file(GLOB DEMO demo/*/*.cpp)
foreach(FILE ${DEMO})
    # 使用文件名作为可执行文件的名称
    get_filename_component(FILE_DIR ${FILE} DIRECTORY)
    get_filename_component(EXECUTABLE_NAME ${FILE_DIR} NAME)
    add_executable(${EXECUTABLE_NAME} ${FILE} ${MODULES})
endforeach()