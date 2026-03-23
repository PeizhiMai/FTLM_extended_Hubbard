cmake_minimum_required(VERSION 3.20)
project(ftlm_extended_hubbard LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(FTLM_BUILD_PYTHON "Build pybind11 Python module" ON)
option(FTLM_BUILD_TESTS "Build unit tests" ON)
option(FTLM_BUILD_BENCHMARKS "Build benchmark executables" ON)

if(FTLM_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
  if(APPLE)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c "import platform; print(platform.machine())"
      OUTPUT_VARIABLE _ftlm_py_machine
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(_ftlm_py_machine)
      set(CMAKE_OSX_ARCHITECTURES "${_ftlm_py_machine}" CACHE STRING "" FORCE)
    endif()
  endif()
endif()

add_library(ftlm_core STATIC
  src/hubbard_params.cpp
  src/version.cpp
  src/lattice.cpp
  src/fock_basis.cpp
  src/hubbard_hamiltonian.cpp
  src/hubbard_momentum_action.cpp
  src/lanczos.cpp
  src/ftlm_thermo.cpp
)
target_include_directories(ftlm_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(ftlm_cli src/main_cli.cpp)
target_link_libraries(ftlm_cli PRIVATE ftlm_core)

if(FTLM_BUILD_BENCHMARKS)
  add_executable(ftlm_bench_2x2 benchmarks/bench_2x2.cpp)
  target_link_libraries(ftlm_bench_2x2 PRIVATE ftlm_core)
  add_executable(ftlm_bench_2x2_nmu benchmarks/bench_2x2_nmu.cpp)
  target_link_libraries(ftlm_bench_2x2_nmu PRIVATE ftlm_core)
  add_executable(ftlm_bench_4x2_ftlm benchmarks/bench_4x2_ftlm.cpp)
  target_link_libraries(ftlm_bench_4x2_ftlm PRIVATE ftlm_core)
  add_executable(ftlm_bench_ftlm_nmu_rect benchmarks/bench_ftlm_nmu_rect.cpp)
  target_link_libraries(ftlm_bench_ftlm_nmu_rect PRIVATE ftlm_core)
endif()

if(FTLM_BUILD_PYTHON)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_ftlm_core python/bindings.cpp)
  target_link_libraries(_ftlm_core PRIVATE ftlm_core)
  set(_ftlm_py_pkg ${CMAKE_BINARY_DIR}/python/ftlm)
  set_target_properties(_ftlm_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${_ftlm_py_pkg})
  file(COPY ${CMAKE_SOURCE_DIR}/python/ftlm/__init__.py DESTINATION ${_ftlm_py_pkg})
endif()

if(FTLM_BUILD_TESTS)
  enable_testing()
  add_executable(ftlm_test tests/smoke.cpp)
  target_link_libraries(ftlm_test PRIVATE ftlm_core)
  add_test(NAME smoke COMMAND ftlm_test)

  add_executable(ftlm_test_pipeline tests/test_hubbard_pipeline.cpp)
  target_link_libraries(ftlm_test_pipeline PRIVATE ftlm_core)
  add_test(NAME hubbard_pipeline COMMAND ftlm_test_pipeline)

  add_executable(ftlm_test_symmetry_stage1 tests/test_symmetry_translation_stage1.cpp)
  target_link_libraries(ftlm_test_symmetry_stage1 PRIVATE ftlm_core)
  add_test(NAME symmetry_translation_stage1 COMMAND ftlm_test_symmetry_stage1)

  add_executable(ftlm_test_symmetry_stage2 tests/test_symmetry_stage2.cpp)
  target_link_libraries(ftlm_test_symmetry_stage2 PRIVATE ftlm_core)
  add_test(NAME symmetry_stage2 COMMAND ftlm_test_symmetry_stage2)

  add_executable(ftlm_test_symmetry_stage3 tests/test_symmetry_stage3.cpp)
  target_link_libraries(ftlm_test_symmetry_stage3 PRIVATE ftlm_core)
  add_test(NAME symmetry_stage3 COMMAND ftlm_test_symmetry_stage3)

  add_executable(ftlm_test_hubbard_momentum_action tests/test_hubbard_momentum_action.cpp)
  target_link_libraries(ftlm_test_hubbard_momentum_action PRIVATE ftlm_core)
  add_test(NAME hubbard_momentum_action COMMAND ftlm_test_hubbard_momentum_action)

  add_executable(ftlm_test_lanczos_engine tests/test_lanczos_engine.cpp)
  target_link_libraries(ftlm_test_lanczos_engine PRIVATE ftlm_core)
  add_test(NAME lanczos_engine COMMAND ftlm_test_lanczos_engine)

  add_executable(ftlm_test_ftlm_thermo tests/test_ftlm_thermo.cpp)
  target_link_libraries(ftlm_test_ftlm_thermo PRIVATE ftlm_core)
  add_test(NAME ftlm_thermo COMMAND ftlm_test_ftlm_thermo)
endif()
