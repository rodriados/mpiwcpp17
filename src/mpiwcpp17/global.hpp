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
#include <mpiwcpp17/detail/raii.hpp>

MPIWCPP17_BEGIN_NAMESPACE

/**
 * The public reference to global world communicator instance. This communicator
 * is the basis for all operations between MPI nodes and therefore cannot be changed
 * or altered in any form. It is used as the default communicator for all collective
 * operations, but can be derived into other communicators as needed.
 * @since 1.0
 */
MPIWCPP17_INLINE const auto world = MPI_COMM_WORLD;

/**
 * The public reference to the current process's rank within the world communicator.
 * This is the global rank within the world communicator, and a process might have
 * different ranks on different communicators.
 * @see mpi::communicator::rank
 * @since 1.0
 */
MPIWCPP17_CONSTEXPR const auto& rank = detail::world_t::rank;

/**
 * The public reference to the number of processes within the world communicator.
 * This is total number of different processes to which communications can occur.
 * A different number of processes might be accessible from different communicators.
 * @see mpi::communicator::size
 * @since 1.0
 */
MPIWCPP17_CONSTEXPR const auto& size = detail::world_t::size;

/**
 * Initializes internal MPI machinery and processes communication.
 * @param argc The number of arguments passed to spawning processes.
 * @param argv The list of arguments passed to spawning processes.
 * @param mode The desired process thread-support level.
 * @return The thread-support level provided by MPI.
 */
MPIWCPP17_INLINE auto initialize(
    int *argc, char ***argv
  , support::thread_level_t mode = support::thread_level_t::single
) -> support::thread_level_t {
    if (detail::world_t::initialized() || detail::world_t::finalized())
        throw exception_t("MPI cannot be initialized");
    return detail::world_t::initialize(argc, argv, mode);
}

/**
 * Initializes the internal MPI machinery and processes communication.
 * @param mode The desired process thread-support level.
 * @return The thread-support level provided by MPI.
 */
MPIWCPP17_INLINE auto initialize(
    support::thread_level_t mode = support::thread_level_t::single
) -> support::thread_level_t {
    return initialize(nullptr, nullptr, mode);
}

/**
 * Informs whether MPI is ready and processes communication is possible.
 * @return Is MPI already initialized?
 * @see mpi::initialize
 */
MPIWCPP17_INLINE auto initialized() -> bool
{
    return detail::world_t::initialized();
}

/**
 * Finalizes MPI, cleans-up resources and disallows further process communication.
 * @see mpi::initialize
 */
MPIWCPP17_INLINE void finalize()
{
    if (detail::world_t::finalized())
        throw exception_t("MPI is already finalized");
    detail::raii_t::clear();
    detail::world_t::finalize();
}

/**
 * Checks whether MPI has already been finalized and process communication is closed.
 * @return Is MPI already finalized?
 * @see mpi::finalize
 */
MPIWCPP17_INLINE auto finalized() -> bool
{
    return detail::world_t::finalized();
}

/**
 * Queries the thread-support level provided by MPI to the current execution.
 * @return The provided thread-support level.
 */
MPIWCPP17_INLINE auto thread_level() -> support::thread_level_t
{
    int level;
    guard(MPI_Query_thread(&level));
    return static_cast<support::thread_level_t>(level);
}

/**
 * Forcebly terminates the entire MPI application and informs an error code.
 * @param code The exit code to be returned by the aborting processes.
 */
MPIWCPP17_INLINE void abort(int code = 1)
{
    detail::raii_t::clear();
    guard(MPI_Abort(world, code));
}

MPIWCPP17_END_NAMESPACE
