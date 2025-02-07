# A thin C++17 wrapper for MPI.
# @file Target declaration for packing the library source code.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2025-present Rodrigo Siqueira
cmake_minimum_required(VERSION 3.19)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(PACK_SCRIPT_SOURCE "support/pack.py")
set(PACK_CONFIGFILE ".packconfig")
set(PACK_OUTPUTFILE "${CMAKE_BINARY_DIR}/mpiwcpp17.h")

# Adds a new custom target that is responsible for packing the library's source
# code into a single header file. This is done so to support an easier use of the
# library, which can therefore be used as a direct import file from any project.
add_custom_target(
  MPIwCpp17PackSourceCode
    COMMAND ${Python3_EXECUTABLE} ${PACK_SCRIPT_SOURCE} -c ${PACK_CONFIGFILE} -o ${PACK_OUTPUTFILE}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    BYPRODUCTS ${PACK_OUTPUTFILE}
    VERBATIM
)
