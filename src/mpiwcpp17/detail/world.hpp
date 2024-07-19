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
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/support.hpp>

MPIWCPP17_BEGIN_NAMESPACE

MPIWCPP17_INLINE auto initialize(int*, char***, support::thread_t) -> support::thread_t;
MPIWCPP17_INLINE void finalize();

namespace detail
{
    /**
     * A wrapper for the globally available world communicator. This wrapper protects
     * the world-communicator so that it can only be instantiated and modified when
     * initializing or finalizing the global MPI state.
     * @since 3.0
     */
    class world_t : public communicator_t
    {
        public:
            process_t rank = process::root;
            int32_t size = 0;

        public:
            MPIWCPP17_CONSTEXPR world_t() noexcept = default;

        private:
            /**
             * Instantiates the globally available world communicator.
             * @see mpi::initialize
             */
            MPIWCPP17_INLINE explicit world_t(int)
              : communicator_t (MPI_COMM_WORLD)
              , rank (mpiwcpp17::rank(MPI_COMM_WORLD))
              , size (mpiwcpp17::size(MPI_COMM_WORLD))
            {}

            using communicator_t::operator=;

        friend auto mpiwcpp17::initialize(int*, char***, support::thread_t) -> support::thread_t;
        friend void mpiwcpp17::finalize();
    };

    /**
     * The world communicator instance. This instance should not be directly used
     * by any user or method. Use the `global` namespace instead.
     * @see mpi::world
     * @since 3.0
     */
    MPIWCPP17_INLINE static detail::world_t world;
}

MPIWCPP17_END_NAMESPACE
