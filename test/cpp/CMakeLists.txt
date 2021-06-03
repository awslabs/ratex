cmake_minimum_required(VERSION 3.0)

set(GTEST_DIR "${CMAKE_BINARY_DIR}/gtest")

get_filename_component(PTMNM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
get_filename_component(PT_DIR "${PTMNM_DIR}/.." ABSOLUTE)
set(MNMDIR "${PTMNM_DIR}/third_party/meta")

file(GLOB PTMNM_LIBDIRS "${PTMNM_DIR}/build/lib.*")
list(GET PTMNM_LIBDIRS 0 PTMNM_LIBDIR)
message("Selected PT/XLA library folder ${PTMNM_LIBDIR}")

file(GLOB LAZYC_LIBDIRS "${PT_DIR}/lazy_tensor_core/build/lib.*")
list(GET LAZYC_LIBDIRS 0 LAZYC_LIBDIR)

execute_process(COMMAND
  python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
  OUTPUT_VARIABLE PT_CXX_ABI
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

project(ptmnm_test)

find_package(PythonLibs)
set(Torch_DIR "${PT_DIR}/torch/share/cmake/Torch")
find_package(Torch REQUIRED)

include(ExternalProject)
set_directory_properties(PROPERTIES EP_PREFIX "${GTEST_DIR}")

ExternalProject_Add(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG 6f5fd0d7199b9a19faa
  SOURCE_DIR "${GTEST_DIR}/src/googletest-src"
  BINARY_DIR "${GTEST_DIR}/src/googletest-build"
  # Disable install step
  INSTALL_COMMAND ""
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  CMAKE_ARGS
    "-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${PT_CXX_ABI}")

ExternalProject_Get_Property(googletest SOURCE_DIR)

set(TORCH_MNM_TEST_SOURCES
  main.cpp
  cpp_test_util.cpp
  test_aten_mnm_tensor.cpp
  torch_mnm_test.cpp
)

add_executable(test_ptmnm ${TORCH_MNM_TEST_SOURCES})

set(TGT_OPTS
  -D_GLIBCXX_USE_CXX11_ABI=${PT_CXX_ABI}
  -Wno-sign-compare
  -Wno-deprecated-declarations
  -Wno-return-type
)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # The -fsized-deallocation is required for Clang to prevent an error on
  # pytorch pybind11 about the operator delete being called inappropriately.
  list(APPEND TGT_OPTS
    -Wno-macro-redefined
    -Wno-return-std-move
    -fsized-deallocation)
endif()

target_compile_options(test_ptmnm PRIVATE ${TGT_OPTS})

target_include_directories(
  test_ptmnm
  PRIVATE
  "${PTMNM_DIR}"
  "${PT_DIR}/lazy_tensor_core"
)
target_include_directories(
  test_ptmnm
  SYSTEM PUBLIC
  "${SOURCE_DIR}/googletest/include"
  "${PYTHON_INCLUDE_DIR}"
)

add_dependencies(test_ptmnm googletest)

ExternalProject_Get_Property(googletest BINARY_DIR)

file(GLOB MNMC_LIBS "${PTXLA_LIBDIR}/_LAZYMNMC.*.so")
list(GET MNMC_LIBS 0 MNMC_LIBRARY)
message("Selected MNMC library ${MNMC_LIBRARY}")

# The linker does not like the _LAZYXLAC.cpython-36m-x86_64-linux-gnu.so name.
execute_process(COMMAND "ln" "-s" "-f"
  "${MNMC_LIBRARY}"
  "${PTXLA_LIBDIR}/libptmnm.so")

find_library(PTMNM_LIB "libptxla.so"
  HINTS "${PTXLA_LIBDIR}")
find_library(PTLTC_LIB "libptltc.so"
  HINTS "${LAZYC_LIBDIR}")
find_library(PTPY_LIB "libtorch_python.so"
  HINTS "${PT_DIR}/torch/lib")
find_library(CLIENT_LIB "libmnm.so"
  HINTS "${PTXLA_DIR}/torch_mnm/lib")
  find_library(CLIENT_LIB "libtvm.so"
  HINTS "${PTXLA_DIR}/torch_mnm/lib")
  find_library(CLIENT_LIB "libtvm_runtime.so"
  HINTS "${PTXLA_DIR}/torch_mnm/lib")

# Use --unresolved-symbols=ignore-all to get around the c10::Half::from_bits
# undefined symbol error at link time. At runtime everything resolves correctly.
target_link_libraries(
  test_ptmnm
  -Wl,--unresolved-symbols=ignore-in-shared-libs
  "${TORCH_LIBRARIES}"
  "${CLIENT_LIB}"
  "${PTMNM_LIB}"
  "${PTLTC_LIB}"
  "${PTPY_LIB}"
  "${BINARY_DIR}/lib/${CMAKE_FIND_LIBRARY_PREFIXES}gtest.a"
  "${PYTHON_LIBRARY}"
  -lutil
  -pthread
  -lstdc++
  -ldl)
