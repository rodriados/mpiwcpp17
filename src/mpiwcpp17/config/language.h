/**
 * A thin C++17 wrapper for MPI.
 * @file Language dialect-specific configurations and macro definitions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

/*
 * Discovers the C++ language dialect in use for the current compilation. A specific
 * dialect might not be supported or might be required for certain functionalities
 * to work properly.
 */
#if defined(__cplusplus)
  #if __cplusplus < 201103L
    #define MPIWCPP17_CPP_DIALECT 2003
  #elif __cplusplus < 201402L
    #define MPIWCPP17_CPP_DIALECT 2011
  #elif __cplusplus < 201703L
    #define MPIWCPP17_CPP_DIALECT 2014
  #elif __cplusplus == 201703L
    #define MPIWCPP17_CPP_DIALECT 2017
  #elif __cplusplus > 201703L
    #define MPIWCPP17_CPP_DIALECT 2020
  #endif
#endif

/*
 * Warns the user if compilation is using a language level lower to the one required
 * by the library. We just emit warning, but allow the compilation to keep going.
 */
#if MPIWCPP17_CPP_DIALECT < 2017
  #warning The MPIwCPP17 library requires C++17.
#endif
