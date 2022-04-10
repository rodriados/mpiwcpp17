/**
 * A thin C++17 wrapper for MPI.
 * @file MPI process identifiers and global value definitions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace process
{
    /**
     * The type for identifying a specific MPI-process.
     * @since 1.0
     */
    using rank = decltype(MPI_ANY_SOURCE);

    /**
     * Defines a special process identifier that may represent any process.
     * @since 1.0
     */
    inline constexpr process::rank any = MPI_ANY_SOURCE;

    /**
     * Defines a special process identifier to indicate that some communication
     * must not perform any effect.
     * @since 1.0
     */
    inline constexpr process::rank null = MPI_PROC_NULL;

    /**
     * Defines a special process identifier to indicate the root of a communicator.
     * @since 1.0
     */
    enum : process::rank { root = 0 };
}

MPIWCPP17_END_NAMESPACE
