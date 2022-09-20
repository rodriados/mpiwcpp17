/**
 * A thin C++17 wrapper for MPI.
 * @file Miscellaneous utilities and global MPI functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <array>
#include <cstdint>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/datatype.hpp>
#include <mpiwcpp17/functor.hpp>
#include <mpiwcpp17/guard.hpp>

#include <mpiwcpp17/detail/communicator/world.hpp>

MPIWCPP17_BEGIN_NAMESPACE

/**
 * The level of MPI thread support.
 * @since 1.0
 */
enum class thread_support : int32_t
{
    single     = MPI_THREAD_SINGLE
  , funneled   = MPI_THREAD_FUNNELED
  , serialized = MPI_THREAD_SERIALIZED
  , multiple   = MPI_THREAD_MULTIPLE
};

/*
 * Forward declaration of initialization and finalization rountines for the global
 * MPI state and processes communication.
 */
inline auto initialize(int*, char***, thread_support = thread_support::single) -> thread_support;
inline auto initialized() -> bool;
inline auto thread_mode() -> thread_support;
inline auto finalize() -> void;
inline auto finalized() -> bool;

namespace detail::world
{
    /**
     * A wrapper for the globally available world-communicator. The wrapper exposes
     * the world-communicator through a const-qualified reference, so that it can
     * only be modified when initializing the global MPI state.
     * @since 1.0
     */
    class communicator
    {
        private:
            inline static mpiwcpp17::detail::communicator::world s_comm;

        public:
            inline static constexpr const mpiwcpp17::detail::communicator::world& ref = s_comm;

        friend auto mpiwcpp17::initialize(int*, char***, thread_support) -> thread_support;
        friend auto mpiwcpp17::finalize() -> void;
    };

    /**
     * The list of deferred functions to call on MPI finalization time. These functions
     * are responsible for releasing acquired resources before finalizing.
     * @see mpi::finalize
     */
    inline static constexpr const auto deferred = std::array {
        &mpiwcpp17::datatype::descriptor::destroy
      , &mpiwcpp17::functor::registry::destroy
    };
}

/**
 * The public reference to the global world-communicator instance.
 * @since 1.0
 */
inline constexpr const detail::communicator::world& world =
    detail::world::communicator::ref;

namespace global
{
    /**
     * The public reference to the current process's rank within world-communicator.
     * @since 1.0
     */
    inline constexpr const process::rank& rank = world.rank;

    /**
     * The public reference to number of processes within the world-communicator.
     * @since 1.0
     */
    inline constexpr const int32_t& size = world.size;
}

/**
 * Initializes the internal MPI state and processes communication.
 * @param mode The desired process thread support level.
 * @return The provided thread support level.
 */
inline auto initialize(thread_support mode = thread_support::single) -> thread_support
{
    return initialize(nullptr, nullptr, mode);
}

/**
 * Initializes the internal MPI state and processes communication.
 * @param argc The number of arguments received via command-line.
 * @param argv The program's command-line arguments.
 * @param mode The desired process thread support level.
 * @return The provided thread support level.
 */
inline auto initialize(int *argc, char ***argv, thread_support mode) -> thread_support
{
    if (int m; !initialized()) {
        guard(MPI_Init_thread(argc, argv, static_cast<int>(mode), &m));
        new (&detail::world::communicator::s_comm) detail::communicator::world (0);
        return static_cast<thread_support>(m);
    } else {
        return thread_mode();
    }
}

/**
 * Checks whether the MPI global state and processes communication has been initialized.
 * @return Was MPI already initialized?
 * @see mpiwcpp17::initialize
 */
inline auto initialized() -> bool
{
    int x; guard(MPI_Initialized(&x));
    return (bool) x;
}

/**
 * Forcebly terminates the entire MPI application abruptly informing an error code.
 * @param code The exit code to be returned by each aborting process.
 */
inline void abort(int code = 1)
{
    for (auto& deferred : detail::world::deferred) { deferred(); }
    guard(MPI_Abort(world, code));
}

/**
 * Queries the thread support level provided by the current MPI execution.
 * @return The provided thread support level.
 */
inline auto thread_mode() -> thread_support
{
    int m; guard(MPI_Query_thread(&m));
    return static_cast<thread_support>(m);
}

/**
 * Terminates MPI execution, cleans up all MPI state and closes processes communication.
 * @see mpiwcpp17::initialize
 */
inline auto finalize() -> void
{
    if (!finalized()) {
        for (auto& deferred : detail::world::deferred) { deferred(); }
        detail::world::communicator::s_comm.~world();
        guard(MPI_Finalize());
    }
}

/**
 * Checks whether the MPI state and processes communication has already been finalized.
 * @return Was MPI already finalized?
 * @see mpiwcpp17::finalize
 */
inline auto finalized() -> bool
{
    int x; guard(MPI_Finalized(&x));
    return (bool) x;
}

MPIWCPP17_END_NAMESPACE
