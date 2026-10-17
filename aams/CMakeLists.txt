cmake_minimum_required(VERSION 3.20)
project(AAMS LANGUAGES CXX)

# AAMS —— Abaqus AMS（PAMS）的逆向复原 C++ 实现（独立构建）。
# 算法/参数语义依据：docs/abaqus_ams/AMS求解流程与参数机制逆向分析.md [R]
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# 依赖（与上游 AMLS 工程一致：MSYS2 Eigen/OpenBLAS/netlib LAPACK +
# 工程内预编译 METIS/GKlib；spdlog 日志）
# ---------------------------------------------------------------------------
find_package(Eigen3 REQUIRED CONFIG)
find_package(spdlog REQUIRED CONFIG)

find_library(OPENBLAS_LIB NAMES openblas libopenblas
             PATHS "D:/msys2/mingw64/lib" REQUIRED)

# netlib LAPACK 静态库：dsygvd/dsygvx 无可变全局状态，可在线程池并发调用。
option(AAMS_STATIC_LAPACK "Use static NETLIB LAPACK with adaptive OpenBLAS BLAS3 wrappers" ON)
if(MINGW AND AAMS_STATIC_LAPACK)
  find_file(NETLIB_LAPACK_STATIC NAMES liblapack.a
            PATHS "D:/msys2/mingw64/lib" REQUIRED NO_DEFAULT_PATH)
  set(NETLIB_LAPACK ${NETLIB_LAPACK_STATIC})
else()
  find_library(NETLIB_LAPACK NAMES lapack liblapack REQUIRED)
endif()

set(THIRD_PARTY_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../third_party)
set(METIS_ROOT ${THIRD_PARTY_ROOT}/metis)
set(GKLIB_ROOT ${THIRD_PARTY_ROOT}/GKlib)
set(METIS_INCLUDE ${METIS_ROOT}/build/xinclude)
find_library(METIS_LIB NAMES metis libmetis
             PATHS ${METIS_ROOT}/build/libmetis REQUIRED)
find_library(GKLIB_LIB NAMES GKlib libGKlib
             PATHS ${GKLIB_ROOT}/build REQUIRED)

# ---------------------------------------------------------------------------
# 库
# ---------------------------------------------------------------------------
file(GLOB_RECURSE AAMS_SOURCES CONFIGURE_DEPENDS src/*.cpp)
list(FILTER AAMS_SOURCES EXCLUDE REGEX "src/apps/")
if(NOT (MINGW AND AAMS_STATIC_LAPACK))
  list(FILTER AAMS_SOURCES EXCLUDE REGEX "src/linalg/lapack_blas\\.cpp$")
endif()

add_library(aams_lib STATIC ${AAMS_SOURCES})
target_include_directories(aams_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${METIS_INCLUDE}
    "D:/msys2/mingw64/include/eigen3"
)
target_link_libraries(aams_lib PUBLIC
    ${NETLIB_LAPACK} ${OPENBLAS_LIB} ${METIS_LIB} ${GKLIB_LIB} spdlog::spdlog
)
if(MINGW)
  target_link_libraries(aams_lib PUBLIC gfortran m gomp)
endif()
if(MINGW AND AAMS_STATIC_LAPACK)
  # 强制从静态归档提取被 wrap 的 LAPACK 引用（与上游 AMLS 同配置）。
  target_link_options(aams_lib INTERFACE
      "LINKER:--wrap=dgemm_,--wrap=dsyrk_,--wrap=dsyr2k_,--wrap=dtrsm_,--wrap=dtrmm_,--wrap=dsymm_,--wrap=dgemmtr_"
      "LINKER:--undefined=__wrap_dgemm_,--undefined=__wrap_dsyrk_,--undefined=__wrap_dsyr2k_,--undefined=__wrap_dtrsm_,--undefined=__wrap_dtrmm_,--undefined=__wrap_dsymm_,--undefined=__wrap_dgemmtr_"
  )
endif()

# ---------------------------------------------------------------------------
# 应用
# ---------------------------------------------------------------------------
add_executable(aams_solve apps/aams_solve.cpp)
target_link_libraries(aams_solve PRIVATE aams_lib)
