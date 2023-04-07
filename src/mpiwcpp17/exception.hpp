/**
 * A thin C++17 wrapper for MPI.
 * @file MPI wrapper exceptions base and common type.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <exception>

#include <mpiwcpp17/environment.hpp>

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
        inline exception_t() = delete;
        inline exception_t(const exception_t&) = default;
        inline exception_t(exception_t&&) = default;

        /**
         * Builds a new exception instance.
         * @param msg The exception's error message.
         */
        inline explicit exception_t(const std::string& msg)
          : m_msg {msg}
        {}

        inline virtual ~exception_t() noexcept = default;

        inline exception_t& operator=(const exception_t&) = delete;
        inline exception_t& operator=(exception_t&&) = delete;

        /**
         * Returns the exception's explanatory string.
         * @return The exception message.
         */
        inline virtual const char *what() const noexcept
        {
            return m_msg.c_str();
        }
};

MPIWCPP17_END_NAMESPACE
