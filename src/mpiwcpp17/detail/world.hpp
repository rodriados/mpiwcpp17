/**
 * A thin C++17 wrapper for MPI.
 * @file The wrapper for the world MPI communicator.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>
#include <cstdint>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/support.hpp>
#include <mpiwcpp17/guard.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail::world
{
    /**
     * The rank of the current process according to the world communicator.
     * This variable should only be changed during MPI initialization.
     * @since 2.1
     */
    MPIWCPP17_INLINE process_t rank = process::root;

    /**
     * The total number of processes in the world communicator.
     * This variable should only be changed during MPI initialization.
     * @since 2.1
     */
    MPIWCPP17_INLINE int32_t size = 0;

    /**
     * Initializes MPI, configures error handling and set world communicator variables.
     * @param argc The number of arguments passed to spawning processes.
     * @param argv The list of arguments passed to spawning processes.
     * @param mode The desired process thread-support level.
     * @return The thread-support level provided by MPI.
     */
    MPIWCPP17_INLINE auto initialize(int *argc, char ***argv, support::thread_level_t mode)
    -> support::thread_level_t {
        int provided = static_cast<int>(mode);
        guard(MPI_Init_thread(argc, argv, provided, &provided));
        guard(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
        guard(MPI_Comm_rank(MPI_COMM_WORLD, &detail::world::rank));
        guard(MPI_Comm_size(MPI_COMM_WORLD, &detail::world::size));
        return static_cast<support::thread_level_t>(provided);
    }

    /**
     * Forcebly terminates the MPI application informing an error code.
     * @param code The exit code to be returned by the aborting processes.
     */
    MPIWCPP17_INLINE void abort(int code = 1)
    {
        guard(MPI_Abort(MPI_COMM_WORLD, code));
    }

    /**
     * Finalizes MPI, cleans-up MPI state and closes all processes communication.
     * @see mpi::detail::world::initialize
     */
    MPIWCPP17_INLINE void finalize()
    {
        guard(MPI_Finalize());
    }
}

MPIWCPP17_END_NAMESPACE
