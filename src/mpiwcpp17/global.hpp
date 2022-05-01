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
#include <utility>

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
 * Forward declaration of initialization and finalization rountines.
 */
inline auto init(thread_support = thread_support::single) -> thread_support;
inline auto init(int*, char***, thread_support = thread_support::single) -> thread_support;
inline bool initialized();
inline void finalize();
inline bool finalized();

namespace detail
{
    /**
     * A wrapper for the global, world-communicator. The wrapper exposes the
     * world-communicator through a const-qualified reference, so that it can
     * only be modified one of the declared friend functions.
     * @since 1.0
     */
    class world
    {
        private:
            /**
             * The concrete world-communicator instance.
             * @since 1.0
             */
            inline static mpiwcpp17::communicator concrete;

        public:
            /**
             * The const-qualified reference to the world-communicator instance.
             * @since 1.0
             */
            inline static constexpr const mpiwcpp17::communicator& ref = concrete;

        friend auto mpiwcpp17::init(int*, char***, thread_support) -> thread_support;
        friend void mpiwcpp17::finalize();
    };

    /**
     * The list of deferred functions to call on MPI finalization time. These functions
     * are responsible for releasing acquired resources before finalizing.
     * @see mpi::finalize
     */
    inline static constexpr const auto deferred = std::array {
        &datatype::descriptor::destroy,
        &functor::registry::destroy
    };
}

/**
 * The public reference to the global world-communicator instance.
 * @since 1.0
 */
inline constexpr const communicator& world = detail::world::ref;

/**
 * Initializes the internal MPI machinery and nodes communication.
 * @param required The desired thread support level.
 * @return The provided thread support level.
 */
inline auto init(thread_support required) -> thread_support
{
    return init(nullptr, nullptr, required);
}

/**
 * Initializes the internal MPI machinery and nodes communication.
 * @param argc The number of arguments received via command-line.
 * @param argv The program's command-line arguments.
 * @param required The desired thread support level.
 * @return The provided thread support level.
 */
inline auto init(int *argc, char ***argv, thread_support required) -> thread_support
{
    int32_t provided;

    if (initialized()) guard(MPI_Query_thread(&provided));
    else guard(MPI_Init_thread(argc, argv, static_cast<int32_t>(required), &provided));

    new (&detail::world::concrete) communicator (MPI_COMM_WORLD);
    return static_cast<thread_support>(provided);
}

/**
 * Checks whether the MPI machinery has already been initialized.
 * @return Was MPI already initialized?
 */
inline auto initialized() -> bool
{
    int32_t initialized;
    guard(MPI_Initialized(&initialized));
    return (bool) initialized;
}

/**
 * Forcebly terminates the entire MPI application abruptly informing an error code.
 * @param code The exit code to be returned by each aborting process.
 */
inline void abort(int code = 1)
{
    for (auto& defer : detail::deferred) defer();
    guard(MPI_Abort(world, code));
}

/**
 * Terminates MPI execution and cleans up all MPI state.
 * @see mpi::init
 */
inline void finalize()
{
    if (finalized()) {
        for (auto& defer : detail::deferred) defer();
        detail::world::concrete.~communicator();
        guard(MPI_Finalize());
    }
}

/**
 * Checks whether the MPI machinery has already been finalized.
 * @return Was MPI already finalized?
 */
inline auto finalized() -> bool
{
    int32_t finalized;
    guard(MPI_Finalized(&finalized));
    return (bool) finalized;
}

MPIWCPP17_END_NAMESPACE
