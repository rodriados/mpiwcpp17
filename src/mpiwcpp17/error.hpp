/**
 * A thin C++17 wrapper for MPI.
 * @file MPI error codes and related functionality.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace error
{
    /**
     * The type of a MPI error code.
     * @since 1.0
     */
    using code = decltype(MPI_SUCCESS);

    /**
     * Defines the error code for a successful MPI operation.
     * @since 1.0
     */
    inline constexpr error::code success = MPI_SUCCESS;

    /**
     * Produces an error message explaining an error returned by MPI.
     * @param err The error code to be described.
     * @return The error description.
     */
    inline auto describe(error::code err) noexcept -> std::string
    {
        int length = MPI_MAX_ERROR_STRING;
        char buffer[MPI_MAX_ERROR_STRING];

        return success == MPI_Error_string(err, buffer, &length)
            ? buffer : "error while describing an MPI error code";
    }
}

MPIWCPP17_END_NAMESPACE
