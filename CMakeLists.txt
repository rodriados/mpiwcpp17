# A thin C++17 wrapper for MPI.
# @file CMake configuration file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2025-present Rodrigo Siqueira
cmake_minimum_required(VERSION 3.29)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CMakePackageConfigHelpers)
include(CMakeDependentOption)
include(HelperFunctions)

# Discovers the project's version and makes it readily available in a variable.
# The version is sourced from the code itself.
get_project_version(MPIWCPP17_VERSION "src/mpiwcpp17/version.h")

# Determines whether the current project is the master target or it is being built
# as a dependency, potentially as a thirdparty, to an external project.
if(NOT DEFINED MPIWCPP17_MASTER_PROJECT)
  set(MPIWCPP17_MASTER_PROJECT OFF)
  if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(MPIWCPP17_MASTER_PROJECT ON)
    message(STATUS "Project: MPIwCpp17 ${MPIWCPP17_VERSION}")
    message(STATUS "CMake version: ${CMAKE_VERSION}")
  endif()
endif()

# Command line options to control the conditions and generation of the targets.
# The targets may be enabled on demand, depending on the user or environment needs.
option(MPIWCPP17_INSTALL "Generate the install target." ON)

cmake_dependent_option(MPIWCPP17_DEVELOPMENT_BUILD "Build tests, enable warnings and errors." ON "BUILD_TESTING" OFF)
cmake_dependent_option(MPIWCPP17_BUILD_TESTING "Generate the test targets." ON "BUILD_TESTING" OFF)
cmake_dependent_option(MPIWCPP17_ENABLE_PEDANTIC "Errors on warnings." ON "BUILD_TESTING" OFF)

# Discovers the project's current version and initializes it. The version is retrieved
# from a source file where the version should be explicitly set.
project(
  MPIwCpp17
    LANGUAGES CXX
    VERSION ${MPIWCPP17_VERSION}
    HOMEPAGE_URL "https://github.com/rodriados/mpiwcpp17"
    DESCRIPTION "A thin C++17 wrapper for MPI.")

include(GNUInstallDirs)

# Installation, configuration and targets variables. These variables are used to
# define where the project must be installed on and where to store its config files.
set(MPIWCPP17_CMAKE_CONFIG_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/MPIwCpp17")
set(MPIWCPP17_INCLUDE_DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mpiwcpp17")
set(MPIWCPP17_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(MPIWCPP17_VERSION_FILE "${CMAKE_CURRENT_BINARY_DIR}/MPIwCpp17ConfigVersion.cmake")
set(MPIWCPP17_CONFIG_FILE "${CMAKE_CURRENT_BINARY_DIR}/MPIwCpp17Config.cmake")
set(MPIWCPP17_CONFIG_FILE_TEMPLATE "${CMAKE_CURRENT_LIST_DIR}/cmake/MPIwCpp17Config.cmake.in")
set(MPIWCPP17_TARGETS_EXPORT_NAME "MPIwCpp17Targets")

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
endif()

# Include required third-party dependencies. These packages are required for the
# functioning of the current project. Therefore, we must bail out if not found.
find_package(MPI REQUIRED)

# Determine whether optional external or third-party dependencies shall be used
# for the current project. Dependencies may be skipped if they are not needed.
if(DEFINED MPIWCPP17_AVOID_DEPENDENCIES OR DEFINED MPIWCPP17_AVOID_THIRDPARTY)
  set(MPIWCPP17_AVOID_THIRDPARTY_REFLECTOR ON)
endif()

# As we define our library as header-only, we must allow all header files to be
# found and eventually exported and installed in the user machine.
add_library(MPIwCpp17 INTERFACE)
add_library(MPIwCpp17::MPIwCpp17 ALIAS MPIwCpp17)

target_compile_features(MPIwCpp17 INTERFACE cxx_std_17)
target_include_directories(
  MPIwCpp17 INTERFACE
    $<BUILD_INTERFACE:${MPIWCPP17_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(MPIwCpp17 INTERFACE MPI::MPI_CXX)

if(NOT DEFINED MPIWCPP17_AVOID_THIRDPARTY_REFLECTOR)
  find_package(Reflector REQUIRED)
  target_link_libraries(MPIwCpp17 INTERFACE Reflector)
endif()

# Sets the compiler flags for a pedantic compilation. We mostly assume that the
# used compiler is GNU-compatible, although some special cases might be cared for.
if(MPIWCPP17_ENABLE_PEDANTIC)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-pedantic-errors -Wall -Wextra -pedantic -Werror)
  elseif(MSVC)
    add_compile_options(/W3 /WX)
  endif()
endif()


# Declaration of target for bundling and packing the project's source code to make
# it possible to include the whole library as a single header-only file.
if(MPIWCPP17_MASTER_PROJECT)
  add_subdirectory(src)
endif()

# Installs the library by copying all needed files to their corresponding destination
# folders. The library should be installed globally in the user system.
if(MPIWCPP17_INSTALL)
  install(
    TARGETS MPIwCpp17
    EXPORT MPIwCpp17Targets
    RUNTIME_DEPENDENCY_SET MPIwCpp17Dependencies
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR})

  export(
    TARGETS MPIwCpp17
    NAMESPACE MPIwCpp17::
    FILE "${PROJECT_BINARY_DIR}/MPIwCpp17Targets.cmake")

  install(
    EXPORT MPIwCpp17Targets
    NAMESPACE MPIwCpp17::
    DESTINATION ${MPIWCPP17_CMAKE_CONFIG_DESTINATION})

  install(
    DIRECTORY "${MPIWCPP17_SOURCE_DIR}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

  configure_package_config_file(
    ${MPIWCPP17_CONFIG_FILE_TEMPLATE}
    ${MPIWCPP17_CONFIG_FILE}
    INSTALL_DESTINATION ${MPIWCPP17_CMAKE_CONFIG_DESTINATION})

  write_basic_package_version_file(
    ${MPIWCPP17_VERSION_FILE}
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

  install(
    FILES
      ${MPIWCPP17_CONFIG_FILE}
      ${MPIWCPP17_VERSION_FILE}
    DESTINATION ${MPIWCPP17_CMAKE_CONFIG_DESTINATION})
endif()

# Declaration of targets for testing. These targets are only available when this
# project is root, therefore not used as dependency for any other.
if(BUILD_TESTING AND MPIWCPP17_MASTER_PROJECT AND MPIWCPP17_BUILD_TESTING)
  include(CTest)
  enable_testing()
  add_subdirectory(test)
endif()
