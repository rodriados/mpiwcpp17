/**
 * A thin C++17 wrapper for MPI.
 * @file Environment configuration and macro values.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpiwcpp17/version.h>

/*
 * Enumerates all possible target environment modes to which the code might be compiled
 * to. The environment mode may affect some features' availability and performace.
 */
#define MPIWCPP17_BUILD_DEV        0
#define MPIWCPP17_BUILD_DEBUG      1
#define MPIWCPP17_BUILD_TESTING    2
#define MPIWCPP17_BUILD_PRODUCTION 3

/*
 * Discovers and explicits the target environment mode to which the code must be
 * currently compiled to. The mode may affect some features' availability and performance.
 */
#if defined(DEBUG) || defined(_DEBUG)
  #define MPIWCPP17_BUILD MPIWCPP17_BUILD_DEBUG
  #define MPIWCPP17_ENVIRONMENT "Debug"
#elif defined(TESTING)
  #define MPIWCPP17_BUILD MPIWCPP17_BUILD_TESTING
  #define MPIWCPP17_ENVIRONMENT "Testing"
#elif defined(DEV) || defined(DEVELOPMENT)
  #define MPIWCPP17_BUILD MPIWCPP17_BUILD_DEV
  #define MPIWCPP17_ENVIRONMENT "Development"
#else
  #define MPIWCPP17_BUILD MPIWCPP17_BUILD_PRODUCTION
  #define MPIWCPP17_ENVIRONMENT "Production"
#endif

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
      MPIWCPP17_EMIT_PRAGMA_CALL(clang warning msg)

  #elif (MPIWCPP17_COMPILER == MPIWCPP17_OPT_COMPILER_MSVC \
      || MPIWCPP17_COMPILER == MPIWCPP17_OPT_COMPILER_ICC)
    #define MPIWCPP17_EMIT_COMPILER_WARNING(msg) \
      MPIWCPP17_EMIT_PRAGMA_CALL(message(msg))

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
#if !defined(MPIWCPP17_IGNORE_CPP_DIALECT)
  #if !defined(MPIWCPP17_CPP_DIALECT) || MPIWCPP17_CPP_DIALECT < 2017
    #warning The MPIwCPP17 library requires at least a C++17 enabled compiler.
  #endif
#endif

/*
 * Macros for applying annotations and qualifiers to functions and methods. As the
 * minimum required language version is C++17, we assume it is guaranteed that all
 * compilers will have `inline` and `constexpr` implemented.
 */
#define MPIWCPP17_INLINE inline
#define MPIWCPP17_CONSTEXPR MPIWCPP17_INLINE constexpr

/**
 * Defines the namespace in which the library lives. This might be overriden if
 * the default namespace value is already in use.
 * @since 1.0
 */
#if defined(MPIWCPP17_OVERRIDE_NAMESPACE)
  #define MPIWCPP17_NAMESPACE MPIWCPP17_OVERRIDE_NAMESPACE
#else
  #define MPIWCPP17_NAMESPACE mpi
#endif

/**
 * This macro is used to open the MPIwCPP17 namespace block and may be overriden
 * if the user so wish or the `mpi::` namespace for some reason already exists.
 * @since 1.0
 */
#define MPIWCPP17_BEGIN_NAMESPACE   \
    namespace MPIWCPP17_NAMESPACE { \
        inline namespace v2 {       \
            namespace mpiwcpp17 = MPIWCPP17_NAMESPACE;

/**
 * This macro is used to close the MPIwCPP17 namespace block and must not be in
 * any way overriden.
 * @since 1.0
 */
#define MPIWCPP17_END_NAMESPACE     }}
