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
#include <mpiwcpp17/guard.hpp>

MPIWCPP17_BEGIN_NAMESPACE

MPIWCPP17_INLINE auto initialize(int*, char***, support::thread_t) -> support::thread_t;
MPIWCPP17_INLINE void finalize();

namespace detail
{
    /**
     * The world communicator instance. This instance should not be directly used
     * by any user or method. Use the `global` namespace instead.
     * @see mpi::world
     * @since 3.0
     */
    MPIWCPP17_INLINE class world_t
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
              : rank (mpiwcpp17::rank(MPI_COMM_WORLD))
              , size (mpiwcpp17::size(MPI_COMM_WORLD))
            {
                guard(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
            }

        friend auto mpiwcpp17::initialize(int*, char***, support::thread_t) -> support::thread_t;
        friend void mpiwcpp17::finalize();
    } world;
}

MPIWCPP17_END_NAMESPACE
