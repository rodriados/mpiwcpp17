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

namespace status
{
    /**
     * The raw MPI status type. This type is responsible for representing the status
     * of a reception or a non-blocking operation.
     * @since 1.0
     */
    using raw = MPI_Status;

    /**
     * The status of the last executed operation.
     * @since 1.0
     */
    inline static status::raw last;

    /**
     * Retrieves the error code of an operation status.
     * @param raw The target operation status instance.
     * @return The MPI operation status error code.
     */
    inline auto error(const status::raw& raw = status::last) noexcept -> error::code
    {
        return raw.MPI_ERROR;
    }

    /**
     * Retrieves the source process of an operation status.
     * @param raw The target operation status instance.
     * @return The MPI operation's source process.
     */
    inline auto source(const status::raw& raw = status::last) noexcept -> process::rank
    {
        return raw.MPI_SOURCE;
    }

    /**
     * Retrieves the message tag of an operation status.
     * @param raw The target operation status instance.
     * @return The MPI operation's message tag.
     */
    inline auto tag(const status::raw& raw = status::last) noexcept -> tag::id
    {
        return raw.MPI_TAG;
    }

    /**
     * Retrieves the number of elements within the message of an operation.
     * @param type The operation's message type identifier.
     * @param raw The target operation status instance.
     * @return The number of elements within operation's message.
     */
    inline auto count(const datatype::id& type, const status::raw& raw = status::last) -> int32_t
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
    inline auto count(const status::raw& raw = status::last) -> int32_t
    {
        return status::count(datatype::identify<T>(), raw);
    }

    /**
     * Determines whether an operation has been cancelled.
     * @param raw The target operation status instance.
     * @return Has the message been cancelled?
     */
    inline auto cancelled(const status::raw& raw) -> bool
    {
        int flag; guard(MPI_Test_cancelled(&raw, &flag));
        return (flag != 0);
    }
}

MPIWCPP17_END_NAMESPACE
