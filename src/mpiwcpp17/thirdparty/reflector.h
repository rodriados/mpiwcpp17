/**
 * A thin C++17 wrapper for MPI.
 * @file The include file for reflector thirdparty library
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpiwcpp17/environment.h>

/*
 * Determines features that must be disabled depending on the current language version
 * available or compiler used for compilation.
 */
#if MPIWCPP17_CPP_DIALECT < 2014                            \
 || (MPIWCPP17_COMPILER != MPIWCPP17_OPT_COMPILER_GCC       \
 &&  MPIWCPP17_COMPILER != MPIWCPP17_OPT_COMPILER_CLANG)
  #ifndef MPIWCPP17_AVOID_THIRDPARTY_REFLECTOR
    #define MPIWCPP17_AVOID_THIRDPARTY_REFLECTOR
  #endif
#endif

#ifdef MPIWCPP17_AVOID_REFLECTOR
  #ifndef MPIWCPP17_AVOID_THIRDPARTY_REFLECTOR
    #define MPIWCPP17_AVOID_THIRDPARTY_REFLECTOR
  #endif
#endif

#if !defined(MPIWCPP17_AVOID_THIRDPARTY) && !defined(MPIWCPP17_AVOID_THIRDPARTY_REFLECTOR)
  #ifdef MPIWCPP17_OVERRIDE_REFLECTOR
    #include MPIWCPP17_OVERRIDE_REFLECTOR
  #elif __has_include(<rodriados/reflector.h>)
    #include <rodriados/reflector.h>
  #elif __has_include(<reflector/api.h>)
    #include <reflector/api.h>
  #else
    #include <reflector.h>
  #endif
#endif
