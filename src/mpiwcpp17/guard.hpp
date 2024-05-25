/**
 * A thin C++17 wrapper for MPI.
 * @file A guard for asserting valid execution states.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <utility>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/exception.hpp>
#include <mpiwcpp17/error.hpp>

MPIWCPP17_DISABLE_GCC_WARNING_BEGIN("-Wattributes")

/*
 * Creates an annotation for an signalling a cold-path to be taken by an if-statement.
 * As we're asserting whether a pre-condition is met or not, we should always favor
 * the likelihood that the condition is indeed met. But if something unexpected
 * unfortunately happens, we can pay the higher cost for messing with the processor's
 * branch predictions in exchange to a slight performance improvement when everything
 * goes as expected, which corresponds to the great majority of times.
 */
#if defined(__has_cpp_attribute) && __has_cpp_attribute(unlikely)
  #define MPIWCPP17_UNLIKELY(condition) \
    ((condition)) [[unlikely]]
#elif (MPIWCPP17_COMPILER == MPIWCPP17_OPT_COMPILER_GCC \
    || MPIWCPP17_COMPILER == MPIWCPP17_OPT_COMPILER_CLANG)
  #define MPIWCPP17_UNLIKELY(condition) \
    (__builtin_expect((condition), 0))
#else
  #define MPIWCPP17_UNLIKELY(condition) \
    ((condition))
#endif

MPIWCPP17_BEGIN_NAMESPACE

/**
 * Asserts whether a MPI call has returned a success error code, and throws an exception
 * otherwise. This function acts just like an assertion, but throwing our own exception.
 * @note The name `assert` is not used because it is reserved by some compilers.
 * @tparam E The exception type to be raised in case of error.
 * @param code The error code returned by the MPI call.
 */
template <typename E = mpiwcpp17::exception_t>
MPIWCPP17_CONSTEXPR void guard(error_t err)
{
    static_assert(std::is_base_of<mpiwcpp17::exception_t, E>::value
      , "only mpiwcpp17 exceptions are throwable from a guard");

  #if !defined(MPIWCPP17_AVOID_GUARD)
    if MPIWCPP17_UNLIKELY (err != error::success) {
        throw E (error::describe(err));
    }
  #endif
}

MPIWCPP17_END_NAMESPACE

#undef MPIWCPP17_UNLIKELY
MPIWCPP17_DISABLE_GCC_WARNING_END("-Wattributes")
