/**
 * A thin C++17 wrapper for MPI.
 * @file MPI error codes and related functionality.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.h>

MPIWCPP17_BEGIN_NAMESPACE

/**
 * The type of a MPI error code.
 * This represents the code of an error returned natively by MPI.
 * @since 1.0
 */
using error_t = decltype(MPI_SUCCESS);

namespace error
{
    enum : error_t
    {
        /**
         * The error code for a successful MPI operation.
         * When returned by MPI, it is a guarantee that the returning operation
         * was successful and that the global MPI state is healthy.
         * @since 1.0
         */
        success = MPI_SUCCESS
    };

    /**
     * Produces an error message explaining an error returned by MPI.
     * @param err The error code to be described.
     * @return The error description.
     */
    MPIWCPP17_INLINE auto describe(error_t err) noexcept -> std::string
    {
        int length = MPI_MAX_ERROR_STRING;
        char buffer[MPI_MAX_ERROR_STRING];

        return success == MPI_Error_string(err, buffer, &length)
            ? buffer : "error while describing an MPI error code";
    }
}

MPIWCPP17_END_NAMESPACE
