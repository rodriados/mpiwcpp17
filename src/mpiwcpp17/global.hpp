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
#include <mpiwcpp17/exception.hpp>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/datatype.hpp>

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
inline void finalize();

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
            inline static mpiw::communicator concrete;

        public:
            /**
             * The const-qualified reference to the world-communicator instance.
             * @since 1.0
             */
            inline static constexpr const mpiw::communicator& ref = concrete;

        friend auto mpiw::init(int*, char***, thread_support) -> thread_support;
        friend void mpiw::finalize();
    };

    /**
     * The list of deferred functions to call on MPI finalization time. These functions
     * are responsible for releasing acquired resources before finalizing.
     * @see mpi::finalize
     */
    inline static constexpr const auto deferred = std::array {
        &datatype::descriptor::destroy,
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
    verify(MPI_Init_thread(argc, argv, static_cast<int32_t>(required), &provided));
    new (&detail::world::concrete) communicator (MPI_COMM_WORLD);
    return static_cast<thread_support>(provided);
}

/**
 * Terminates MPI execution and cleans up all MPI state.
 * @see mpi::init
 */
inline void finalize()
{
    for (auto& defer : detail::deferred) defer();
    detail::world::concrete.~communicator();
    verify(MPI_Finalize());
}

MPIWCPP17_END_NAMESPACE
