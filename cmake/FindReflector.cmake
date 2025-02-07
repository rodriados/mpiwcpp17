# A thin C++17 wrapper for MPI.
# @file Script responsible for finding the Reflector project.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2025-present Rodrigo Siqueira
cmake_minimum_required(VERSION 3.24)

include(FetchContent)

set(REFLECTOR_REPOSITORY "https://github.com/rodriados/reflector.git")
set(REFLECTOR_REPOSITORY_TAG "v1.0.0")

# Declares the remote source of the required package and allows it to be found.
# If needed, the package will be downloaded and cached for build.
FetchContent_Declare(
  Reflector
    GIT_SHALLOW true
    GIT_REPOSITORY ${REFLECTOR_REPOSITORY}
    GIT_TAG ${REFLECTOR_REPOSITORY_TAG}
    OVERRIDE_FIND_PACKAGE)

# Now that the package is declared, we must find and configure it so that its variables
# and targets are made available for the parent context.
find_package(Reflector REQUIRED)
