/**
 * A thin C++17 wrapper for MPI.
 * @file The global MPI references.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>
#include <cstdint>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/process.hpp>

#include <mpiwcpp17/detail/world.hpp>

MPIWCPP17_BEGIN_NAMESPACE

/**
 * The public reference to global world communicator instance. This communicator
 * is the basis for all operations between MPI nodes and therefore cannot be changed
 * or altered in any form. It is used as the default communicator for all collective
 * operations, but can be derived into other communicators as needed.
 * @since 1.0
 */
MPIWCPP17_INLINE const communicator_t world = MPI_COMM_WORLD;

namespace global
{
    /**
     * The public reference to the current process's rank within the world communicator.
     * This is the global rank within the world communicator, and a process might
     * have different ranks on different communicators.
     * @see mpi::rank
     * @since 1.0
     */
    MPIWCPP17_CONSTEXPR const process_t& rank = detail::world.rank;

    /**
     * The public reference to the number of processes within the world communicator.
     * This is total number of different processes to which communications can occur.
     * A different number of processes might be accessible from different communicators.
     * @see mpi::size
     * @since 1.0
     */
    MPIWCPP17_CONSTEXPR const int32_t& size = detail::world.size;
}

MPIWCPP17_END_NAMESPACE
