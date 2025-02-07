# A thin C++17 wrapper for MPI.
# @file CMake tests configuration file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2025-present Rodrigo Siqueira
cmake_minimum_required(VERSION 3.29)

find_package(Catch2 REQUIRED)

# The test sources should be automatically discovered by iterating over the directory.
# By doing this, we avoid the need to manually add new tests to this file.
file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS *.cpp)

# Configure the test target and makes it available as an executable. This executable
# should be responsible for running all tests listed in the repository.
add_executable(MPIwCpp17Test ${TEST_SOURCES})
target_link_libraries(MPIwCpp17Test PRIVATE Catch2::Catch2 ${PROJECT_NAME})
target_include_directories(MPIwCpp17Test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Sets the MPI emulator script as a property of our test target to allow tests to
# be executed using MPI, instead of running in a single process.
set_target_properties(
  MPIwCpp17Test PROPERTIES
    CROSSCOMPILING_EMULATOR "${CMAKE_CURRENT_SOURCE_DIR}/mpi-emulator.sh")

# Automatically discovers the tests found on the repository and allows them to be
# called from and linked to ctest.
catch_discover_tests(MPIwCpp17Test DISCOVERY_MODE "PRE_TEST" EXTRA_ARGS --order decl)
