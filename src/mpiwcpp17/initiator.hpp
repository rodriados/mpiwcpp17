/**
 * A thin C++17 wrapper for MPI.
 * @file The RAII initiator for global MPI machinery.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/support.hpp>
#include <mpiwcpp17/world.hpp>

#include <mpiwcpp17/detail/deferrer.hpp>

MPIWCPP17_BEGIN_NAMESPACE

MPIWCPP17_INLINE support::thread_t initialize(int*, char***, support::thread_t = support::thread_t::single);
MPIWCPP17_INLINE support::thread_t thread_mode();
MPIWCPP17_INLINE bool initialized();
MPIWCPP17_INLINE void finalize();
MPIWCPP17_INLINE bool finalized();

/**
 * Automatically initializes and finalizes the global MPI state. This type cannot
 * be instatiated more than once.
 * @since 1.0
 */
struct initiator_t final
{
    const support::thread_t thread_mode;

    MPIWCPP17_INLINE initiator_t() noexcept = delete;
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
 * @param mode The desired process thread support level.
 * @return The provided thread support level.
 */
MPIWCPP17_INLINE support::thread_t initialize(support::thread_t mode = support::thread_t::single)
{
    return mpiwcpp17::initialize(nullptr, nullptr, mode);
}

/**
 * Initializes the internal MPI state and processes communication.
 * @param argc The number of arguments received via command-line.
 * @param argv The program's command-line arguments.
 * @param mode The desired process thread support level.
 * @return The provided thread support level.
 */
MPIWCPP17_INLINE support::thread_t initialize(int *argc, char ***argv, support::thread_t mode)
{
    if (int provided; !mpiwcpp17::initialized()) {
        guard(MPI_Init_thread(argc, argv, static_cast<int>(mode), &provided));
        new (&detail::world_t::s_world) detail::world_t::communicator_t (0);
        return static_cast<support::thread_t>(provided);
    } else {
        return thread_mode();
    }
}

/**
 * Initializes the internal MPI state and processes communication.
 * @param mode The desired process thread support level.
 */
MPIWCPP17_INLINE initiator_t::initiator_t(support::thread_t mode)
  : thread_mode (mpiwcpp17::initialize(mode))
{}

/**
 * Initializes the internal MPI state and processes communication with arguments.
 * @param argc The number of command line arguments to initialize MPI with.
 * @param argv The list of processes' command line arguments.
 * @param mode The desired process thread support level.
 */
MPIWCPP17_INLINE initiator_t::initiator_t(int *argc, char ***argv, support::thread_t mode)
  : thread_mode (mpiwcpp17::initialize(argc, argv, mode))
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
    detail::deferrer_t::run();
    guard(MPI_Abort(world, code));
}

/**
 * Terminates MPI execution, cleans up all MPI state and closes processes communication.
 * @see mpi::initialize
 */
MPIWCPP17_INLINE void finalize()
{
    if (!mpiwcpp17::finalized()) {
        detail::deferrer_t::run();
        new (&detail::world_t::s_world) communicator_t ();
        guard(MPI_Finalize());
    }
}

/**
 * Checks whether the MPI state and processes communication has already been finalized.
 * @return Was MPI already finalized?
 * @see mpi::finalize
 */
MPIWCPP17_INLINE auto finalized() -> bool
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
    mpiwcpp17::finalize();
}

MPIWCPP17_END_NAMESPACE
