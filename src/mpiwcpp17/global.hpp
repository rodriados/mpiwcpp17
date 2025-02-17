/**
 * A thin C++17 wrapper for MPI.
 * @file The global MPI references.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/exception.hpp>
#include <mpiwcpp17/support.hpp>
#include <mpiwcpp17/guard.hpp>

#include <mpiwcpp17/detail/world.hpp>
#include <mpiwcpp17/detail/tracker.hpp>

MPIWCPP17_BEGIN_NAMESPACE

/**
 * The public reference to global world communicator instance. This communicator
 * is the basis for all operations between MPI nodes and therefore cannot be changed
 * or altered in any form. It is used as the default communicator for all collective
 * operations, but can be derived into other communicators as needed.
 * @since 1.0
 */
MPIWCPP17_INLINE const auto world = MPI_COMM_WORLD;

namespace global
{
    /**
     * The public reference to the current process's rank within the world communicator.
     * This is the global rank within the world communicator, and a process might
     * have different ranks on different communicators.
     * @see mpi::communicator::rank
     * @since 1.0
     */
    MPIWCPP17_CONSTEXPR const auto& rank = detail::world::rank;

    /**
     * The public reference to the number of processes within the world communicator.
     * This is total number of different processes to which communications can occur.
     * A different number of processes might be accessible from different communicators.
     * @see mpi::communicator::size
     * @since 1.0
     */
    MPIWCPP17_CONSTEXPR const auto& size = detail::world::size;
}

MPIWCPP17_INLINE auto initialized() -> bool;
MPIWCPP17_INLINE auto finalized() -> bool;

/**
 * Initializes internal MPI machinery and processes communication.
 * @param argc The number of arguments passed to spawning processes.
 * @param argv The list of arguments passed to spawning processes.
 * @param mode The desired process thread-support level.
 * @return The thread-support level provided by MPI.
 */
MPIWCPP17_INLINE support::thread_level_t initialize(
    int *argc, char ***argv
  , support::thread_level_t mode = support::thread_level_t::single
) {
    if (initialized() || finalized())
        throw exception_t("MPI is already initialized or finalized");
    return detail::world::initialize(argc, argv, mode);
}

/**
 * Initializes the internal MPI machinery and processes communication.
 * @param mode The desired process thread-support level.
 * @return The thread-support level provided by MPI.
 */
MPIWCPP17_INLINE support::thread_level_t initialize(
    support::thread_level_t mode = support::thread_level_t::single
) {
    return initialize(nullptr, nullptr, mode);
}

/**
 * Finalizes MPI, cleans-up resources and closes all processes communication.
 * @see mpi::initialize
 */
MPIWCPP17_INLINE void finalize()
{
    if (finalized())
        throw exception_t("MPI is already finalized");
    detail::tracker_t::clear();
    detail::world::finalize();
}

/**
 * Checks whether MPI is ready and processes communication has been opened.
 * @return Is MPI already initialized?
 * @see mpi::initialize
 */
MPIWCPP17_INLINE bool initialized()
{
    int f; guard(MPI_Initialized(&f));
    return (bool) f;
}

/**
 * Checks whether MPI is finalized and processes communication has been closed.
 * @return Is MPI already finalized?
 * @see mpi::finalize
 */
MPIWCPP17_INLINE bool finalized()
{
    int f; guard(MPI_Finalized(&f));
    return (bool) f;
}

/**
 * Queries the thread-support level provided by MPI to the current execution.
 * @return The provided thread-support level.
 */
MPIWCPP17_INLINE support::thread_level_t thread_level()
{
    int t; guard(MPI_Query_thread(&t));
    return static_cast<support::thread_level_t>(t);
}

/**
 * Forcebly terminates the entire MPI application and informs an error code.
 * @param code The exit code to be returned by the aborting processes.
 */
MPIWCPP17_INLINE void abort(int code = 1)
{
    detail::tracker_t::clear();
    detail::world::abort(code);
}

MPIWCPP17_END_NAMESPACE
