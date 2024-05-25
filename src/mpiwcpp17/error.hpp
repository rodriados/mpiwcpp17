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

namespace error
{
    /**
     * The type of a MPI error code.
     * @since 1.0
     */
    using raw_t = decltype(MPI_SUCCESS);

    /**
     * Defines the error code for a successful MPI operation.
     * @since 1.0
     */
    enum : raw_t { success = MPI_SUCCESS };

    /**
     * Produces an error message explaining an error returned by MPI.
     * @param err The error code to be described.
     * @return The error description.
     */
    MPIWCPP17_INLINE auto describe(raw_t err) noexcept -> std::string
    {
        int length = MPI_MAX_ERROR_STRING;
        char buffer[MPI_MAX_ERROR_STRING];

        return success == MPI_Error_string(err, buffer, &length)
            ? buffer : "error while describing an MPI error code";
    }
}

/**
 * Exposing the raw error type to the project's root namespace, allowing it to be
 * referenced by with decreased verbosity.
 * @since 1.0
 */
using error_t = error::raw_t;

MPIWCPP17_END_NAMESPACE
