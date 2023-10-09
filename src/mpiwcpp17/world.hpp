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
enum class thread_support_t : int32_t
{
    single     = MPI_THREAD_SINGLE
  , funneled   = MPI_THREAD_FUNNELED
  , serialized = MPI_THREAD_SERIALIZED
  , multiple   = MPI_THREAD_MULTIPLE
};

/*#@+
 * Forward declaration of initialization and finalization rountines for the global
 * MPI state and processes communication.
 */
inline auto initialize(int*, char***, thread_support_t = thread_support_t::single) -> thread_support_t;
inline auto initialized() -> bool;
inline auto thread_mode() -> thread_support_t;
inline auto finalize() -> void;
inline auto finalized() -> bool;
/**#@-*/

namespace detail
{
    /**
     * A wrapper for the world-communicator singleton instance. The wrapper exposes
     * the world-communicator through a const-qualified reference, so that it can
     * only be modified within befriended scopes.
     * @since 1.0
     */
    class world_t
    {
        private:
            /**
             * A wrapper for the globally available world communicator. This wrapper protects
             * the world-communicator so that it can only be instantiated and modified when
             * initializing or finalizing the global MPI state.
             * @since 1.0
             */
            struct communicator_t : public mpiwcpp17::communicator_t {
                inline constexpr communicator_t() noexcept = default;

                /**
                 * Instantiates the globally available world communicator.
                 * @see mpi::initialize
                 */
                inline explicit communicator_t(int)
                  : mpiwcpp17::communicator_t (MPI_COMM_WORLD)
                {}

                using mpiwcpp17::communicator_t::operator=;
            };

        private:
            inline static communicator_t s_comm;

        public:
            inline static constexpr const communicator_t& s_ref = s_comm;

        friend auto mpiwcpp17::initialize(int*, char***, thread_support_t) -> thread_support_t;
        friend auto mpiwcpp17::finalize() -> void;
    };

    /**
     * The call to deferred destruction functions to call on MPI finalization time.
     * These functions are responsible for releasing acquired resources before finalizing.
     * @see mpi::finalize
     */
    inline static void deferred_destroy()
    {
        mpiwcpp17::datatype::descriptor_t::destroy();
        mpiwcpp17::functor::registry_t::destroy();
    }
}

/**
 * The public reference to the global world-communicator instance.
 * @since 1.0
 */
inline constexpr const communicator_t& world = detail::world_t::s_ref;

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
inline auto initialize(thread_support_t mode = thread_support_t::single) -> thread_support_t
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
inline auto initialize(int *argc, char ***argv, thread_support_t mode) -> thread_support_t
{
    if (int m; !initialized()) {
        guard(MPI_Init_thread(argc, argv, static_cast<int>(mode), &m));
        new (&detail::world_t::s_comm) detail::world_t::communicator_t (0);
        return static_cast<thread_support_t>(m);
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
    detail::deferred_destroy();
    guard(MPI_Abort(world, code));
}

/**
 * Queries the thread support level provided by the current MPI execution.
 * @return The provided thread support level.
 */
inline auto thread_mode() -> thread_support_t
{
    int m; guard(MPI_Query_thread(&m));
    return static_cast<thread_support_t>(m);
}

/**
 * Terminates MPI execution, cleans up all MPI state and closes processes communication.
 * @see mpi::initialize
 */
inline auto finalize() -> void
{
    if (!finalized()) {
        detail::deferred_destroy();
        new (&detail::world_t::s_comm) communicator_t ();
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
