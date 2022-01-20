/**
 * A thin C++17 wrapper for MPI.
 * @file Environment configuration and macro values.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpiwcpp17/version.h>

#include <mpiwcpp17/config/compiler.h>
#include <mpiwcpp17/config/language.h>
#include <mpiwcpp17/config/namespace.hpp>

/*
 * Determines features that must be disabled depending on the current language version
 * available or compiler used for compilation.
 */
#if MPIWCPP17_CPP_DIALECT < 2014                            \
 || (MPIWCPP17_COMPILER != MPIWCPP17_OPT_COMPILER_GCC       \
 &&  MPIWCPP17_COMPILER != MPIWCPP17_OPT_COMPILER_CLANG)
  #if !defined(MPIWCPP17_AVOID_REFLECTION)
    #define MPIWCPP17_AVOID_REFLECTION
  #endif
#endif
