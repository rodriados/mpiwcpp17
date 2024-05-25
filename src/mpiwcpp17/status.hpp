/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI representation of the status of an operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <cstdint>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/datatype.hpp>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/error.hpp>
#include <mpiwcpp17/guard.hpp>
#include <mpiwcpp17/tag.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace status
{
    /**
     * The raw MPI status type.
     * @since 3.0
     */
    using raw_t = MPI_Status;

    /**
     * Wraps the status of a reception, non-blocking or test operation.
     * @since 1.0
     */
    class wrapper_t
    {
        private:
            raw_t m_status;

        public:
            MPIWCPP17_INLINE wrapper_t() noexcept = default;
            MPIWCPP17_INLINE wrapper_t(const wrapper_t&) noexcept = default;
            MPIWCPP17_INLINE wrapper_t(wrapper_t&&) noexcept = default;

            /**
             * Wraps a native MPI status instance.
             * @param s The status instance to be wrapped.
             */
            MPIWCPP17_INLINE wrapper_t(const raw_t& s) noexcept
              : m_status (s)
            {}

            MPIWCPP17_INLINE wrapper_t& operator=(const wrapper_t&) noexcept = default;
            MPIWCPP17_INLINE wrapper_t& operator=(wrapper_t&&) noexcept = default;

            /**
             * Exposes the wrapped operation status instance, allowing it to be used
             * seamlessly with native MPI functions.
             * @return The wrapped status instance.
             */
            MPIWCPP17_INLINE operator raw_t&() noexcept
            {
                return m_status;
            }

            /**
             * Exposes the pointer to the wrapped operation status instance, allowing
             * the wrapper to be seamlessly used with native MPI functions.
             * @return The wrapped status pointer.
             */
            MPIWCPP17_INLINE operator raw_t*() noexcept
            {
                return &m_status;
            }
    };

    /**
     * The flag for denoting that status should be ignored for a specific operation.
     * This can be used with any native MPI function that outputs a status instance.
     * @since 2.0
     */
    MPIWCPP17_CONSTEXPR raw_t* ignore = MPI_STATUS_IGNORE;

    /**
     * Retrieves the error code of an operation status.
     * @param stt The target operation status instance.
     * @return The MPI operation status error code.
     */
    MPIWCPP17_INLINE error_t error(const raw_t& stt) noexcept
    {
        return stt.MPI_ERROR;
    }

    /**
     * Retrieves the source process of an operation status.
     * @param stt The target operation status instance.
     * @return The MPI operation's source process.
     */
    MPIWCPP17_INLINE process_t source(const raw_t& stt) noexcept
    {
        return stt.MPI_SOURCE;
    }

    /**
     * Retrieves the message tag of an operation status.
     * @param stt The target operation status instance.
     * @return The MPI operation's message tag.
     */
    MPIWCPP17_INLINE tag_t tag(const raw_t& stt) noexcept
    {
        return stt.MPI_TAG;
    }

    /**
     * Retrieves the number of elements within the message of an operation.
     * @param stt The target operation status instance.
     * @param type The operation's message type identifier.
     * @return The number of elements within operation's message.
     */
    MPIWCPP17_INLINE int32_t count(const raw_t& stt, const datatype_t& type)
    {
        int count; guard(MPI_Get_count(&stt, type, &count));
        return count != MPI_UNDEFINED ? count : -1;
    }

    /**
     * Retrieves the number of elements within the message of an operation.
     * @tparam T The operation's message type.
     * @param stt The target operation status instance.
     * @return The number of elements within operation's message.
     */
    template <typename T>
    MPIWCPP17_INLINE int32_t count(const raw_t& stt)
    {
        return count(stt, datatype::identify<T>());
    }

    /**
     * Determines whether an operation has been cancelled.
     * @param stt The target operation status instance.
     * @return Has the message been cancelled?
     */
    MPIWCPP17_INLINE bool cancelled(const raw_t& stt)
    {
        int flag; guard(MPI_Test_cancelled(&stt, &flag));
        return (flag != 0);
    }
}

/**
 * Exposing the status wrapper type to the project's root namespace, allowing it
 * to be referenced by with decreased verbosity.
 * @since 1.0
 */
using status_t = status::wrapper_t;

MPIWCPP17_END_NAMESPACE
