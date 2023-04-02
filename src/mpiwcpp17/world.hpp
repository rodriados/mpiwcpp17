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
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/datatype.hpp>
#include <mpiwcpp17/functor.hpp>
#include <mpiwcpp17/guard.hpp>

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
     * A wrapper for the globally available world communicator. This wrapper protects
     * the world-communicator so that it can only be instantiated and modified when
     * initializing or finalizing the global MPI state.
     * @since 1.0
     */
    struct communicator final : public mpiwcpp17::communicator
    {
        inline constexpr communicator() noexcept = default;

        /**
         * Instantiates the globally available world communicator.
         * @see mpi::initialize
         */
        inline explicit communicator(int)
          : mpiwcpp17::communicator (MPI_COMM_WORLD)
        {}

        using mpiwcpp17::communicator::operator=;
    };

    /**
     * A wrapper for the world-communicator singleton instance. The wrapper exposes
     * the world-communicator through a const-qualified reference, so that it can
     * only be modified within selected scopes.
     * @since 1.0
     */
    class instance
    {
        private:
            inline static detail::world::communicator s_world;

        public:
            inline static constexpr const detail::world::communicator& ref = s_world;

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
inline constexpr const detail::world::communicator& world = detail::world::instance::ref;

namespace global
{
    /**
     * The public reference to the current process's rank within world-communicator.
     * @since 1.0
     */
    inline constexpr const process_t& rank = world.rank;

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
        new (&detail::world::instance::s_world) detail::world::communicator (0);
        return static_cast<thread_support>(m);
    } else {
        return thread_mode();
    }
}

/**
 * Checks whether the MPI global state and processes communication has been initialized.
 * @return Was MPI already initialized?
 * @see mpi::initialize
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
 * @see mpi::initialize
 */
inline auto finalize() -> void
{
    if (!finalized()) {
        for (auto& deferred : detail::world::deferred) { deferred(); }
        detail::world::instance::s_world = std::move(communicator());
        guard(MPI_Finalize());
    }
}

/**
 * Checks whether the MPI state and processes communication has already been finalized.
 * @return Was MPI already finalized?
 * @see mpi::finalize
 */
inline auto finalized() -> bool
{
    int x; guard(MPI_Finalized(&x));
    return (bool) x;
}

MPIWCPP17_END_NAMESPACE
