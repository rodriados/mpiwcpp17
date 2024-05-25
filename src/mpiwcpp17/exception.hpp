/**
 * A thin C++17 wrapper for MPI.
 * @file MPI wrapper exceptions base and common type.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <exception>

#include <mpiwcpp17/environment.h>

MPIWCPP17_BEGIN_NAMESPACE

/**
 * Represents an exception that can be thrown and propagated through the code carrying
 * a message about the MPI error that has been detected.
 * @since 1.0
 */
class exception_t : public std::exception
{
    private:
        const std::string m_msg;

    public:
        MPIWCPP17_INLINE exception_t() = delete;
        MPIWCPP17_INLINE exception_t(const exception_t&) = default;
        MPIWCPP17_INLINE exception_t(exception_t&&) = default;

        /**
         * Builds a new exception instance.
         * @param msg The exception's error message.
         */
        MPIWCPP17_INLINE explicit exception_t(const std::string& msg)
          : m_msg (msg)
        {}

        MPIWCPP17_INLINE virtual ~exception_t() noexcept = default;

        MPIWCPP17_INLINE exception_t& operator=(const exception_t&) = delete;
        MPIWCPP17_INLINE exception_t& operator=(exception_t&&) = delete;

        /**
         * Returns the exception's explanatory string.
         * @return The exception message.
         */
        MPIWCPP17_INLINE virtual const char *what() const noexcept
        {
            return m_msg.c_str();
        }
};

MPIWCPP17_END_NAMESPACE
