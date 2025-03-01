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

namespace detail
{
    /**
     * The global MPI world state. This static object is responsible for initializing
     * and finalizing the MPI machinery in every process execution. Though it should
     * not be called directly by the library's users, this is true starting point
     * to any process that requires MPI communication.
     * @since 2.1
     */
    struct world_t final
    {
        /**
         * The rank of the current process according to the world communicator.
         * This variable should only be changed during MPI initialization.
         * @since 2.1
         */
        MPIWCPP17_INLINE static process_t rank = process::root;

        /**
         * The total number of processes in the world communicator.
         * This variable should only be changed during MPI initialization.
         * @since 2.1
         */
        MPIWCPP17_INLINE static int32_t size = 0;

        /**
         * Initializes MPI, configures error handling and sets world variables.
         * @param argc The number of arguments passed to spawning processes.
         * @param argv The list of arguments passed to spawning processes.
         * @param mode The desired process thread-support level.
         * @return The thread-support level provided by MPI.
         * @see mpi::initialize
         */
        MPIWCPP17_INLINE static auto initialize(int *argc, char ***argv, support::thread_level_t mode)
        -> support::thread_level_t {
            int provided = static_cast<int>(mode);
            guard(MPI_Init_thread(argc, argv, provided, &provided));
            guard(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
            guard(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
            guard(MPI_Comm_size(MPI_COMM_WORLD, &size));
            return static_cast<support::thread_level_t>(provided);
        }

        /**
         * Informs whether MPI has been initialized and is ready, and whether processes
         * communication is open, active and ready.
         * @return Is MPI already initialized?
         * @see mpi::detail::world_t::initialize
         * @see mpi::initialized
         */
        MPIWCPP17_INLINE static auto initialized() -> bool
        {
            int flag;
            guard(MPI_Initialized(&flag));
            return (bool) flag;
        }

        /**
         * Finalizes MPI, cleans-up internal MPI state and closes processes communication.
         * After this function has been executed, no other MPI calls can be performed.
         * @see mpi::detail::world_t::initialize
         * @see mpi::finalize
         */
        MPIWCPP17_INLINE static void finalize()
        {
            guard(MPI_Finalize());
        }

        /**
         * Informs whether MPI has been finalized, therefore disallowing any further
         * communication between processes. If so, MPI is in an invalid state.
         * @return Is MPI already finalized?
         * @see mpi::detail::world_t::finalize
         * @see mpi::finalized
         */
        MPIWCPP17_INLINE static auto finalized() -> bool
        {
            int flag;
            guard(MPI_Finalized(&flag));
            return (bool) flag;
        }
    };
}

MPIWCPP17_END_NAMESPACE
