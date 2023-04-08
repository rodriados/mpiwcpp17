/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI representation of the status of an operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <cstdint>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/datatype.hpp>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/error.hpp>
#include <mpiwcpp17/guard.hpp>
#include <mpiwcpp17/tag.hpp>

MPIWCPP17_BEGIN_NAMESPACE

/**
 * Wraps the status of a reception, non-blocking or test operation.
 * @since 1.0
 */
class status_t
{
    public:
        using raw_t = MPI_Status;

    private:
        raw_t m_status;

    public:
        inline status_t() noexcept = default;
        inline status_t(const status_t&) noexcept = default;
        inline status_t(status_t&&) noexcept = default;

        /**
         * Wraps a native MPI status instance.
         * @param s The status instance to be wrapped.
         */
        inline status_t(const raw_t& s) noexcept
          : m_status (s)
        {}

        inline status_t& operator=(const status_t&) noexcept = default;
        inline status_t& operator=(status_t&&) noexcept = default;

        /**
         * Exposes the wrapped operation status instance, allowing it to be used
         * seamlessly with native MPI functions.
         * @return The wrapped status instance.
         */
        inline operator raw_t&() noexcept
        {
            return m_status;
        }

        /**
         * Exposes the pointer to the wrapped operation status instance, allowing
         * the wrapper to be seamlessly used with native MPI functions.
         * @return The wrapped status pointer.
         */
        inline operator raw_t*() noexcept
        {
            return &m_status;
        }
};

namespace status
{
    /**
     * Retrieves the error code of an operation status.
     * @param s The target operation status instance.
     * @return The MPI operation status error code.
     */
    inline auto error(const status_t::raw_t& s) noexcept -> error_t
    {
        return s.MPI_ERROR;
    }

    /**
     * Retrieves the source process of an operation status.
     * @param s The target operation status instance.
     * @return The MPI operation's source process.
     */
    inline auto source(const status_t::raw_t& s) noexcept -> process_t
    {
        return s.MPI_SOURCE;
    }

    /**
     * Retrieves the message tag of an operation status.
     * @param s The target operation status instance.
     * @return The MPI operation's message tag.
     */
    inline auto tag(const status_t::raw_t& s) noexcept -> tag_t
    {
        return s.MPI_TAG;
    }

    /**
     * Retrieves the number of elements within the message of an operation.
     * @param type The operation's message type identifier.
     * @param s The target operation status instance.
     * @return The number of elements within operation's message.
     */
    inline auto count(const datatype_t& type, const status_t::raw_t& s) -> int32_t
    {
        int count; guard(MPI_Get_count(&s, type, &count));
        return count != MPI_UNDEFINED ? count : -1;
    }

    /**
     * Retrieves the number of elements within the message of an operation.
     * @tparam T The operation's message type.
     * @param s The target operation status instance.
     * @return The number of elements within operation's message.
     */
    template <typename T>
    inline auto count(const status_t::raw_t& s) -> int32_t
    {
        return count(datatype::identify<T>(), s);
    }

    /**
     * Determines whether an operation has been cancelled.
     * @param s The target operation status instance.
     * @return Has the message been cancelled?
     */
    inline auto cancelled(const status_t::raw_t& s) -> bool
    {
        int flag; guard(MPI_Test_cancelled(&s, &flag));
        return (flag != 0);
    }
}

MPIWCPP17_END_NAMESPACE
