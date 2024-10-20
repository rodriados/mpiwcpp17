/**
 * A thin C++17 wrapper for MPI.
 * @file The RAII initiator for global MPI machinery.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/support.hpp>

#include <mpiwcpp17/detail/world.hpp>
#include <mpiwcpp17/detail/tracker.hpp>

MPIWCPP17_BEGIN_NAMESPACE
MPIWCPP17_FWD_GLOBAL_STATUS_FUNCTIONS

MPIWCPP17_INLINE support::thread_t thread_mode();

/**
 * Automatically initializes and finalizes the global MPI state. This type cannot
 * be instatiated more than once.
 * @since 1.0
 */
struct initiator_t final
{
    const support::thread_t thread_mode;

    MPIWCPP17_INLINE initiator_t(const initiator_t&) noexcept = delete;
    MPIWCPP17_INLINE initiator_t(initiator_t&&) noexcept = delete;

    MPIWCPP17_INLINE initiator_t(support::thread_t = support::thread_t::single);
    MPIWCPP17_INLINE initiator_t(int*, char***, support::thread_t = support::thread_t::single);

    MPIWCPP17_INLINE initiator_t& operator=(const initiator_t&) noexcept = delete;
    MPIWCPP17_INLINE initiator_t& operator=(initiator_t&&) noexcept = delete;

    MPIWCPP17_INLINE ~initiator_t();
};

/**
 * Initializes the internal MPI state and processes communication.
 * @param argc The number of arguments received via command-line.
 * @param argv The program's command-line arguments.
 * @param mode The desired process thread support level.
 * @return The provided thread support level.
 */
MPIWCPP17_INLINE support::thread_t initialize(
    int *argc
  , char ***argv
  , support::thread_t mode = support::thread_t::single
) {
    if (int provided; !initialized()) {
        guard(MPI_Init_thread(argc, argv, static_cast<int>(mode), &provided));
        new (&detail::g_world) detail::world_t (0);
        return static_cast<support::thread_t>(provided);
    } else {
        return thread_mode();
    }
}

/**
 * Initializes the internal MPI state and processes communication.
 * @param mode The desired process thread support level.
 * @return The provided thread support level.
 */
MPIWCPP17_INLINE support::thread_t initialize(support::thread_t mode = support::thread_t::single)
{
    return initialize(nullptr, nullptr, mode);
}

/**
 * Initializes the internal MPI state and processes communication.
 * @param mode The desired process thread support level.
 */
MPIWCPP17_INLINE initiator_t::initiator_t(support::thread_t mode)
  : thread_mode (initialize(mode))
{}

/**
 * Initializes the internal MPI state and processes communication with arguments.
 * @param argc The number of command line arguments to initialize MPI with.
 * @param argv The list of processes' command line arguments.
 * @param mode The desired process thread support level.
 */
MPIWCPP17_INLINE initiator_t::initiator_t(int *argc, char ***argv, support::thread_t mode)
  : thread_mode (initialize(argc, argv, mode))
{}

/**
 * Queries the thread support level provided by the current MPI execution.
 * @return The provided thread support level.
 */
MPIWCPP17_INLINE support::thread_t thread_mode()
{
    int provided; guard(MPI_Query_thread(&provided));
    return static_cast<support::thread_t>(provided);
}

/**
 * Checks whether the MPI global state and processes communication has been initialized.
 * @return Was MPI already initialized?
 * @see mpi::initialize
 */
MPIWCPP17_INLINE bool initialized()
{
    int flag; guard(MPI_Initialized(&flag));
    return (bool) flag;
}

/**
 * Forcebly terminates the entire MPI application abruptly informing an error code.
 * @param code The exit code to be returned by each aborting process.
 */
MPIWCPP17_INLINE void abort(int code = 1)
{
    detail::tracker_t::clear();
    guard(MPI_Abort(MPI_COMM_WORLD, code));
}

/**
 * Terminates MPI execution, cleans up all MPI state and closes processes communication.
 * @see mpi::initialize
 */
MPIWCPP17_INLINE void finalize()
{
    if (!finalized()) {
        detail::tracker_t::clear();
        detail::g_world = detail::world_t ();
        guard(MPI_Finalize());
    }
}

/**
 * Checks whether the MPI state and processes communication has already been finalized.
 * @return Was MPI already finalized?
 * @see mpi::finalize
 */
MPIWCPP17_INLINE bool finalized()
{
    int flag; guard(MPI_Finalized(&flag));
    return (bool) flag;
}

/**
 * Terminates the global MPI state and communication between processes.
 * @see mpi::finalize
 */
MPIWCPP17_INLINE initiator_t::~initiator_t()
{
    finalize();
}

MPIWCPP17_END_NAMESPACE
