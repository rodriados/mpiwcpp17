# A thin C++17 wrapper for MPI.
# @file Script responsible for finding the Catch2 project.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2025-present Rodrigo Siqueira
cmake_minimum_required(VERSION 3.24)

include(FetchContent)

set(CATCH2_REPOSITORY "https://github.com/catchorg/Catch2.git")
set(CATCH2_REPOSITORY_TAG "v3.9.1")

# Declares the remote source of the required package and allows it to be found.
# If needed, the package will be downloaded and cached for build.
FetchContent_Declare(
  Catch2
    GIT_SHALLOW true
    GIT_REPOSITORY ${CATCH2_REPOSITORY}
    GIT_TAG ${CATCH2_REPOSITORY_TAG}
    OVERRIDE_FIND_PACKAGE)

# Now that the package is declared, we must find and configure it so that its variables
# and targets are made available for the parent context.
find_package(Catch2 REQUIRED)

# Also, a helper function for automatically finding tests is provided. We must include
# this function as well, to provide a better downstream experience.
include(Catch)
