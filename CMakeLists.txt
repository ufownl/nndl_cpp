cmake_minimum_required(VERSION 2.8)

project(NNDL_CPP)

# silence policy CMP0042 warning by enabling RPATH explicitly
if(APPLE AND NOT DEFINED CMAKE_MACOSX_RPATH)
  set(CMAKE_MACOSX_RPATH TRUE)
endif()

set(CMAKE_MODULE_PATH ${NNDL_CPP_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/bin")
set(LIBRARY_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/lib")
set(NNDL_CPP_3RDPARTY_DIR ${NNDL_CPP_SOURCE_DIR}/3rdparty)
set(BUILD_SHARED_LIBS ON)

include_directories(
  ${NNDL_CPP_3RDPARTY_DIR}/Eigen3/include
  ${NNDL_CPP_SOURCE_DIR}
)

add_subdirectory(utils)
add_subdirectory(chapter1)
