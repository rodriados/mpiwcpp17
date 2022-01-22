/**
 * A thin C++17 wrapper for MPI.
 * @file MPI node process identifiers and global value definitions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace node
{
    /**
     * The type for identifying a specific MPI-node.
     * @since 1.0
     */
    using id = decltype(MPI_ANY_SOURCE);

    /**
     * Defines a special node identifier that may represent any node.
     * @since 1.0
     */
    inline constexpr node::id any = MPI_ANY_SOURCE;

    /**
     * Defines a special node identifier to indicate the root of a communicator.
     * @since 1.0
     */
    inline constexpr node::id root = MPI_ROOT;

    /**
     * Defines a special node identifier to indicate that some communication must
     * not perform any effect.
     * @since 1.0
     */
    inline constexpr node::id null = MPI_PROC_NULL;
}

MPIWCPP17_END_NAMESPACE
