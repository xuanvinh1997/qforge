cmake_minimum_required(VERSION 3.15)
project(qsun_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 REQUIRED)
find_package(OpenMP QUIET)

set(QSUN_SOURCES
    src/state_vector.cpp
    src/gates_single.cpp
    src/gates_controlled.cpp
    src/gates_swap.cpp
    src/gates_noise.cpp
    src/measurement.cpp
    src/data_ops.cpp
)

pybind11_add_module(_qforge_core ${QSUN_SOURCES} bindings/py_qsun.cpp)
target_include_directories(_qforge_core PRIVATE include)

if(OpenMP_CXX_FOUND)
    target_link_libraries(_qforge_core PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_qforge_core PRIVATE QSUN_USE_OPENMP)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(_qforge_core PRIVATE -O3 -march=native -ffast-math)
elseif(MSVC)
    target_compile_options(_qforge_core PRIVATE /O2 /fp:fast)
endif()

install(TARGETS _qforge_core DESTINATION Qsun)
