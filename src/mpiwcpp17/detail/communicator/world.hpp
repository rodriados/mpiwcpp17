/**
 * A thin C++17 wrapper for MPI.
 * @file MPI world communicator wrapper.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/communicator.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail::communicator
{
    /**
     * A wrapper for the globally available world communicator. This wrapper protects
     * the world-communicator so that it can only be instantiated and modified when
     * initializing the global MPI state.
     * @since 1.0
     */
    struct world final : public mpiwcpp17::communicator
    {
        inline constexpr world() noexcept = default;

        /**
         * Instantiates the globally available world communicator.
         * @see mpiwcpp17::initialize
         */
        inline explicit world(int)
          : mpiwcpp17::communicator (MPI_COMM_WORLD)
        {}
    };
}

MPIWCPP17_END_NAMESPACE
