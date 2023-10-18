cmake_minimum_required(VERSION 3.21)

project(MLAS LANGUAGES CXX ASM)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - using Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (UNIX)
  add_library(gemv STATIC src/sgemm.cpp src/linux/SgemmKernelM1Avx.S)
elseif(MSVC)
  enable_language(ASM_MASM)
  add_library(gemv STATIC src/sgemm.cpp src/win/SgemmKernelM1Avx.asm)
endif()

add_executable(run src/main.cc)
target_link_libraries(run PRIVATE gemv)
