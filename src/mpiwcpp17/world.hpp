/**
 * A thin C++17 wrapper for MPI.
 * @file Miscellaneous utilities and global MPI functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <cstdint>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/support.hpp>

MPIWCPP17_BEGIN_NAMESPACE

MPIWCPP17_INLINE support::thread_t initialize(int*, char***, support::thread_t);
MPIWCPP17_INLINE void finalize();

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
                MPIWCPP17_CONSTEXPR communicator_t() noexcept = default;

                /**
                 * Instantiates the globally available world communicator.
                 * @see mpi::initialize
                 */
                MPIWCPP17_INLINE explicit communicator_t(int)
                  : mpiwcpp17::communicator_t (MPI_COMM_WORLD)
                {}

                using mpiwcpp17::communicator_t::operator=;
            };

        private:
            MPIWCPP17_INLINE static communicator_t s_world;

        public:
            MPIWCPP17_CONSTEXPR static const communicator_t& s_worldref = s_world;
            MPIWCPP17_INLINE static process_t s_rank = process::root;
            MPIWCPP17_INLINE static int32_t s_size = 0;

        friend auto mpiwcpp17::initialize(int*, char***, support::thread_t) -> support::thread_t;
        friend void mpiwcpp17::finalize();
    };
}

namespace global
{
    /**
     * The public reference to the current process's rank within world-communicator.
     * @since 1.0
     */
    MPIWCPP17_CONSTEXPR const process_t& rank = detail::world_t::s_rank;

    /**
     * The public reference to number of processes within the world-communicator.
     * @since 1.0
     */
    MPIWCPP17_CONSTEXPR const int32_t& size = detail::world_t::s_size;
}

/**
 * The public reference to the global world-communicator instance.
 * @since 1.0
 */
MPIWCPP17_CONSTEXPR const communicator_t& world = detail::world_t::s_worldref;

MPIWCPP17_END_NAMESPACE
