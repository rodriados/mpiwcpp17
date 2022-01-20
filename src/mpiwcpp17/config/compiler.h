/**
 * A thin C++17 wrapper for MPI.
 * @file Compiler-specific configurations and macro definitions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

/*
 * Enumerating known compilers. These compilers are not all necessarily officially
 * supported. Nevertheless, some special adaptation or fixes might be implemented
 * to each one of these if so needed.
 */
#define MPIWCPP17_OPT_COMPILER_UNKNOWN 0
#define MPIWCPP17_OPT_COMPILER_GCC     1
#define MPIWCPP17_OPT_COMPILER_CLANG   2
#define MPIWCPP17_OPT_COMPILER_MSVC    3
#define MPIWCPP17_OPT_COMPILER_ICC     4

/*
 * Finds the version of the compiler currently in use. Some features might change
 * or be unavailable depending on the compiler configuration.
 */
#if defined(__clang__)
  #define MPIWCPP17_COMPILER MPIWCPP17_OPT_COMPILER_CLANG
  #define MPIWCPP17_CLANG_VERSION (__clang_major__ * 100 + __clang_minor__)
  #define MPIWCPP17_COMPILER_VERSION MPIWCPP17_CLANG_VERSION

#elif defined(__INTEL_COMPILER)
  #define MPIWCPP17_COMPILER MPIWCPP17_OPT_COMPILER_ICC
  #define MPIWCPP17_ICC_VERSION __INTEL_COMPILER
  #define MPIWCPP17_COMPILER_VERSION MPIWCPP17_ICC_VERSION

#elif defined(_MSC_VER)
  #define MPIWCPP17_COMPILER MPIWCPP17_OPT_COMPILER_MSVC
  #define MPIWCPP17_MSVC_VERSION _MSC_VER
  #define MPIWCPP17_COMPILER_VERSION MPIWCPP17_MSVC_VERSION

#elif defined(__GNUC__)
  #define MPIWCPP17_COMPILER MPIWCPP17_OPT_COMPILER_GCC
  #define MPIWCPP17_GCC_VERSION (__GNUC__ * 100 + __GUNC_MINOR__)
  #define MPIWCPP17_COMPILER_VERSION MPIWCPP17_GCC_VERSION

#else
  #define MPIWCPP17_COMPILER MPIWCPP17_OPT_COMPILER_UNKNOWN
  #define MPIWCPP17_COMPILER_VERSION 0
#endif

/*
 * Macro for programmatically emitting a pragma call, independently on which compiler
 * is currently in use.
 */
#if !defined(MPIWCPP17_EMIT_PRAGMA_CALL)
  #define MPIWCPP17_EMIT_PRAGMA_CALL(x) _Pragma(#x)
#endif

/*
 * Macro for programmatically emitting a compiler warning. If active, this will
 * be shown during compile-time and might stop compilation if warnings are errors.
 */
#if !defined(MPIWCPP17_EMIT_COMPILER_WARNING)
  #if (MPIWCPP17_COMPILER == MPIWCPP17_OPT_COMPILER_GCC)
    #define MPIWCPP17_EMIT_COMPILER_WARNING(msg) \
      MPIWCPP17_EMIT_PRAGMA_CALL(GCC warning msg)

  #elif (MPIWCPP17_COMPILER == MPIWCPP17_OPT_COMPILER_CLANG)
    #define MPIWCPP17_EMIT_COMPILER_WARNING(msg) \
      MPIWCPP17_EMIT_COMPILER_WARNING(clang warning msg)

  #elif (MPIWCPP17_COMPILER == MPIWCPP17_OPT_COMPILER_MSVC \
      || MPIWCPP17_COMPILER == MPIWCPP17_OPT_COMPILER_ICC)
    #define MPIWCPP17_EMIT_COMPILER_WARNING(msg) \
      MPIWCPP17_EMIT_COMPILER_WARNING(message(msg))

  #else
      #define MPIWCPP17_EMIT_COMPILER_WARNING(msg)
  #endif
#endif

/*
 * Macros for disabling or manually emitting warnings with specific compilers. This
 * is useful to treat the behaviour of a specific compiler, such as hiding buggy
 * compiler warnings or exploits that have intentionally been taken advantage of.
 */
#if (MPIWCPP17_COMPILER == MPIWCPP17_OPT_COMPILER_GCC)
  #define MPIWCPP17_EMIT_GCC_WARNING(x) MPIWCPP17_EMIT_COMPILER_WARNING(x)
  #define MPIWCPP17_DISABLE_GCC_WARNING_BEGIN(x)            \
    MPIWCPP17_EMIT_PRAGMA_CALL(GCC diagnostic push)         \
    MPIWCPP17_EMIT_PRAGMA_CALL(GCC diagnostic ignored x)
  #define MPIWCPP17_DISABLE_GCC_WARNING_END(x)              \
    MPIWCPP17_EMIT_PRAGMA_CALL(GCC diagnostic pop)
#else
  #define MPIWCPP17_EMIT_GCC_WARNING(x)
  #define MPIWCPP17_DISABLE_GCC_WARNING_BEGIN(x)
  #define MPIWCPP17_DISABLE_GCC_WARNING_END(x)
#endif

#if (MPIWCPP17_COMPILER == MPIWCPP17_OPT_COMPILER_CLANG)
  #define MPIWCPP17_EMIT_CLANG_WARNING(x) MPIWCPP17_EMIT_COMPILER_WARNING(x)
  #define MPIWCPP17_DISABLE_CLANG_WARNING_BEGIN(x)          \
    MPIWCPP17_EMIT_PRAGMA_CALL(clang diagnostic push)       \
    MPIWCPP17_EMIT_PRAGMA_CALL(clang diagnostic ignored x)
  #define MPIWCPP17_DISABLE_CLANG_WARNING_END(x)            \
    MPIWCPP17_EMIT_PRAGMA_CALL(clang diagnostic pop)
#else
  #define MPIWCPP17_EMIT_CLANG_WARNING(x)
  #define MPIWCPP17_DISABLE_CLANG_WARNING_BEGIN(x)
  #define MPIWCPP17_DISABLE_CLANG_WARNING_END(x)
#endif

#if (MPIWCPP17_COMPILER == MPIWCPP17_OPT_COMPILER_MSVC)
  #define MPIWCPP17_EMIT_MSVC_WARNING(x) MPIWCPP17_EMIT_COMPILER_WARNING(x)
  #define MPIWCPP17_DISABLE_MSVC_WARNING_BEGIN(x)           \
    MPIWCPP17_EMIT_PRAGMA_CALL(warning(push))               \
    MPIWCPP17_EMIT_PRAGMA_CALL(warning(disable : x))
  #define MPIWCPP17_DISABLE_MSVC_WARNING_END(x)             \
    MPIWCPP17_EMIT_PRAGMA_CALL(warning(pop))
#else
  #define MPIWCPP17_EMIT_MSVC_WARNING(x)
  #define MPIWCPP17_DISABLE_MSVC_WARNING_BEGIN(x)
  #define MPIWCPP17_DISABLE_MSVC_WARNING_END(x)
#endif

#if (MPIWCPP17_COMPILER == MPIWCPP17_OPT_COMPILER_ICC)
  #define MPIWCPP17_EMIT_ICC_WARNING(x) MPIWCPP17_EMIT_COMPILER_WARNING(x)
  #define MPIWCPP17_DISABLE_ICC_WARNING_BEGIN(x)            \
    MPIWCPP17_EMIT_PRAGMA_CALL(warning(push))               \
    MPIWCPP17_EMIT_PRAGMA_CALL(warning(disable : x))
  #define MPIWCPP17_DISABLE_ICC_WARNING_END(x)              \
    MPIWCPP17_EMIT_PRAGMA_CALL(warning(pop))
#else
  #define MPIWCPP17_EMIT_ICC_WARNING(x)
  #define MPIWCPP17_DISABLE_ICC_WARNING_BEGIN(x)
  #define MPIWCPP17_DISABLE_ICC_WARNING_END(x)
#endif
