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
 * Wraps the status of a reception or asynchronous non-blocking operation.
 * @since 1.0
 */
class status
{
    public:
        using raw_type = MPI_Status;

    private:
        raw_type m_status;

    public:
        inline status() noexcept = default;
        inline status(const status&) noexcept = default;
        inline status(status&&) noexcept = default;

        /**
         * Wraps a native MPI status instance.
         * @param s The status instance to be wrapped.
         */
        inline status(raw_type s) noexcept
          : m_status (s)
        {}

        inline status& operator=(const status&) noexcept = default;
        inline status& operator=(status&&) noexcept = default;

        /**
         * Exposes the wrapped operation status instance, allowing it to be used
         * seamlessly with native MPI functions.
         * @return The wrapped status instance.
         */
        inline operator raw_type&() noexcept
        {
            return m_status;
        }

        /**
         * Exposes the pointer to the wrapped operation status instance, allowing
         * the wrapper to be seamlessly used with native MPI functions.
         * @return The wrapped status pointer.
         */
        inline operator raw_type*() noexcept
        {
            return &m_status;
        }

    public:
        /**
         * Retrieves the error code of an operation status.
         * @param raw The target operation status instance.
         * @return The MPI operation status error code.
         */
        inline static auto error(const status::raw_type& raw) noexcept -> error::code
        {
            return raw.MPI_ERROR;
        }

        /**
         * Retrieves the source process of an operation status.
         * @param raw The target operation status instance.
         * @return The MPI operation's source process.
         */
        inline static auto source(const status::raw_type& raw) noexcept -> process::rank
        {
            return raw.MPI_SOURCE;
        }

        /**
         * Retrieves the message tag of an operation status.
         * @param raw The target operation status instance.
         * @return The MPI operation's message tag.
         */
        inline static auto tag(const status::raw_type& raw) noexcept -> tag::id
        {
            return raw.MPI_TAG;
        }

        /**
         * Retrieves the number of elements within the message of an operation.
         * @param type The operation's message type identifier.
         * @param raw The target operation status instance.
         * @return The number of elements within operation's message.
         */
        inline static auto count(const datatype::id& type, const status::raw_type& raw) -> int32_t
        {
            int count; guard(MPI_Get_count(&raw, type, &count));
            return count != MPI_UNDEFINED ? count : -1;
        }

        /**
         * Retrieves the number of elements within the message of an operation.
         * @tparam T The operation's message type.
         * @param raw The target operation status instance.
         * @return The number of elements within operation's message.
         */
        template <typename T>
        inline static auto count(const status::raw_type& raw) -> int32_t
        {
            return status::count(datatype::identify<T>(), raw);
        }

        /**
         * Determines whether an operation has been cancelled.
         * @param raw The target operation status instance.
         * @return Has the message been cancelled?
         */
        inline static auto cancelled(const status::raw_type& raw) -> bool
        {
            int flag; guard(MPI_Test_cancelled(&raw, &flag));
            return (flag != 0);
        }
};

MPIWCPP17_END_NAMESPACE
