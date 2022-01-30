/**
 * A thin C++17 wrapper for MPI.
 * @file MPI exceptions and assertions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <string>
#include <utility>
#include <exception>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/error.hpp>

MPIWCPP17_BEGIN_NAMESPACE

/**
 * Represents an exception that can be thrown and propagated through the code carrying
 * a message about the MPI error that has been detected.
 * @since 1.0
 */
class exception : public std::exception
{
    private:
        const std::string m_msg;

    public:
        inline exception() = delete;
        inline exception(const exception&) = default;
        inline exception(exception&&) = default;

        /**
         * Builds a new exception instance.
         * @param msg The exception's error message.
         */
        inline explicit exception(const std::string& msg)
          : m_msg {msg}
        {}

        inline virtual ~exception() noexcept = default;

        inline exception& operator=(const exception&) = delete;
        inline exception& operator=(exception&&) = delete;

        /**
         * Returns the exception's explanatory string.
         * @return The exception message.
         */
        inline virtual const char *what() const noexcept
        {
            return m_msg.c_str();
        }
};

MPIWCPP17_DISABLE_GCC_WARNING_BEGIN("-Wattributes")

/*
 * Creates an annotation for an signalling a cold-path to be taken by an if-statement.
 * As we're asserting whether a pre-condition is met or not, we should always favor
 * the likelihood that the condition is indeed met. But if something unexpected
 * unfortunately happens, we can pay the higher cost for messing with the processor's
 * branch predictions in exchange to a slight performance improvement when everything
 * goes as expected, the great majority of times.
 */
#if defined(__has_cpp_attribute) && __has_cpp_attribute(unlikely)
  #define __mpiwcpp17_unlikely__(condition) \
    ((condition)) [[unlikely]]
#elif (MPIWCPP17_COMPILER == MPIWCPP17_OPT_COMPILER_GCC \
    || MPIWCPP17_COMPILER == MPIWCPP17_OPT_COMPILER_CLANG)
  #define __mpiwcpp17_unlikely__(condition) \
    (__builtin_expect((condition), 0))
#else
  #define __mpiwcpp17_unlikely__(condition) \
    ((condition))
#endif

/**
 * Asserts whether a MPI call has returned a success error code, and throws an exception
 * otherwise. This function acts just like an assertion, but throwing our own exception.
 * @note The name `assert` is not used because it is reserved by some compilers.
 * @tparam E The exception type to be raised in case of error.
 * @param code The error code returned by the MPI call.
 */
template <typename E = exception>
inline constexpr void verify(error::code code)
{
    static_assert(std::is_base_of<exception, E>::value, "only exception types are throwable");
    if __mpiwcpp17_unlikely__ (code != MPI_SUCCESS) {
        throw E (error::describe(code));
    }
}

#undef __mpiwcpp17_unlikely__
MPIWCPP17_DISABLE_GCC_WARNING_END("-Wattributes")

MPIWCPP17_END_NAMESPACE
