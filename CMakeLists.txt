cmake_minimum_required(VERSION 3.20)
project(AMLS LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
# Eigen3 (header-only)
find_package(Eigen3 REQUIRED CONFIG)

# OpenBLAS（BLAS + cblas，OpenMP 版仅用于 gemm；并发 LAPACK 走 netlib）
find_library(OPENBLAS_LIB NAMES openblas libopenblas
             PATHS "D:/msys2/mingw64/lib" REQUIRED)

# netlib LAPACK（静态）：dsygvd/dsygvx 等驱动例程无可变全局状态，
# 可在线程池中并发调用（OpenMP 版 OpenBLAS 的并发 LAPACK 会互踩，实测慢 9 倍）。
# 链接顺序必须 LAPACK 在前：其 BLAS 符号由 OpenBLAS 提供。
find_library(NETLIB_LAPACK NAMES lapack liblapack
             PATHS "D:/msys2/mingw64/lib" REQUIRED)

# Catch2 v3
find_package(Catch2 3 REQUIRED CONFIG)

# SuiteSparse CHOLMOD (预留, v1 不链接)
# find_library(CHOLMOD_LIB NAMES cholmod PATHS "D:/msys2/mingw64/lib")

# METIS + GKlib (工程内预编译静态库)
set(METIS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/metis)
set(GKLIB_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/GKlib)
set(METIS_INCLUDE ${METIS_ROOT}/build/xinclude)
find_library(METIS_LIB NAMES metis libmetis
             PATHS ${METIS_ROOT}/build/libmetis REQUIRED)
find_library(GKLIB_LIB NAMES GKlib libGKlib
             PATHS ${GKLIB_ROOT}/build REQUIRED)

# Spectra (header-only, 参考解)
set(SPECTRA_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/spectra/include)

# spdlog（日志；MSYS2 包，fmt 已随包安装）
find_package(spdlog REQUIRED CONFIG)

# ---------------------------------------------------------------------------
# 库
# ---------------------------------------------------------------------------
file(GLOB_RECURSE AMLS_SOURCES CONFIGURE_DEPENDS src/*.cpp)
list(FILTER AMLS_SOURCES EXCLUDE REGEX "src/apps/")

add_library(amls_lib STATIC ${AMLS_SOURCES})
target_include_directories(amls_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${METIS_INCLUDE}
    ${SPECTRA_INCLUDE}
    "D:/msys2/mingw64/include/eigen3"
)
target_link_libraries(amls_lib PUBLIC
    ${NETLIB_LAPACK} ${OPENBLAS_LIB} ${METIS_LIB} ${GKLIB_LIB} spdlog::spdlog
)
if(MINGW)
  target_link_libraries(amls_lib PUBLIC gfortran m gomp)
endif()

# ---------------------------------------------------------------------------
# 应用
# ---------------------------------------------------------------------------
add_executable(amls_solve src/apps/amls_solve.cpp)
target_link_libraries(amls_solve PRIVATE amls_lib)

add_executable(amls_verify src/apps/amls_verify.cpp)
target_link_libraries(amls_verify PRIVATE amls_lib)

add_executable(reference_tool src/apps/reference_tool.cpp)
target_link_libraries(reference_tool PRIVATE amls_lib)

# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------
enable_testing()

file(GLOB UNIT_TESTS CONFIGURE_DEPENDS tests/unit/*.cpp)
if(UNIT_TESTS)
  add_executable(unit_tests ${UNIT_TESTS})
  target_link_libraries(unit_tests PRIVATE amls_lib Catch2::Catch2WithMain)
  add_test(NAME unit COMMAND unit_tests)
endif()

file(GLOB REGRESSION_TESTS CONFIGURE_DEPENDS tests/regression/*.cpp)
if(REGRESSION_TESTS)
  add_executable(regression_tests ${REGRESSION_TESTS})
  target_link_libraries(regression_tests PRIVATE amls_lib Catch2::Catch2WithMain)
  add_test(NAME regression COMMAND regression_tests)
endif()
