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
  #ifndef MPIWCPP17_AVOID_REFLECTION
    #define MPIWCPP17_AVOID_REFLECTION
  #endif
#endif

#ifdef MPIWCPP17_AVOID_REFLECTION
  #ifndef REFLECTOR_AVOID_LOOPHOLE
    #define REFLECTOR_AVOID_LOOPHOLE
  #endif
#endif

#ifndef MPIWCPP17_AVOID_INCLUDE_THIRDPARTY
  #ifdef MPIWCPP17_OVERRIDE_REFLECTOR
    #include MPIWCPP17_OVERRIDE_REFLECTOR
  #elif __has_include(<reflector/api.h>)
    #include <reflector/api.h>
  #elif __has_include(<reflector.h>)
    #include <reflector.h>
  #elifndef MPIWCPP17_AVOID_THIRDPARTY_REFLECTOR
    #define MPIWCPP17_AVOID_THIRDPARTY_REFLECTOR
  #endif
#endif
