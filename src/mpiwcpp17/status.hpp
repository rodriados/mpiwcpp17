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

/**
 * The raw MPI status type.
 * @since 3.0
 */
using status_t = MPI_Status;

namespace status
{
    /**
     * The flag for denoting that status should be ignored for a specific operation.
     * This can be used with any native MPI function that outputs a status instance.
     * @since 2.0
     */
    MPIWCPP17_INLINE status_t* ignore = MPI_STATUS_IGNORE;

    /**
     * Retrieves the error code of an operation status.
     * @param stt The target operation status instance.
     * @return The MPI operation status error code.
     */
    MPIWCPP17_INLINE error_t error(const status_t& stt) noexcept
    {
        return stt.MPI_ERROR;
    }

    /**
     * Retrieves the source process of an operation status.
     * @param stt The target operation status instance.
     * @return The MPI operation's source process.
     */
    MPIWCPP17_INLINE process_t source(const status_t& stt) noexcept
    {
        return stt.MPI_SOURCE;
    }

    /**
     * Retrieves the message tag of an operation status.
     * @param stt The target operation status instance.
     * @return The MPI operation's message tag.
     */
    MPIWCPP17_INLINE tag_t tag(const status_t& stt) noexcept
    {
        return stt.MPI_TAG;
    }

    /**
     * Retrieves the number of elements within the message of an operation.
     * @param stt The target operation status instance.
     * @param type The operation's message type identifier.
     * @return The number of elements within operation's message.
     */
    MPIWCPP17_INLINE int32_t count(const status_t& stt, const datatype_t& type)
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
    MPIWCPP17_INLINE int32_t count(const status_t& stt)
    {
        return count(stt, datatype::identify<T>());
    }

    /**
     * Determines whether an operation has been cancelled.
     * @param stt The target operation status instance.
     * @return Has the message been cancelled?
     */
    MPIWCPP17_INLINE bool cancelled(const status_t& stt)
    {
        int flag; guard(MPI_Test_cancelled(&stt, &flag));
        return (flag != 0);
    }
}

MPIWCPP17_END_NAMESPACE
