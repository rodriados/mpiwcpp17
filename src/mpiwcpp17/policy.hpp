/**
 * A thin C++17 wrapper for MPI.
 * @file Policies for miscellaneous behaviour guarantees.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2025-present Rodrigo Siqueira
 */
#pragma once

#include <mpiwcpp17/environment.h>

MPIWCPP17_BEGIN_NAMESPACE

namespace policy
{
    inline namespace operation
    {
        /**
         * Guarantees that the quantity of message elements is uniform across all
         * processes taking part in an operation, either by sending or receiving
         * the same number of elements.
         * @since 1.0
         */
        MPIWCPP17_CONSTEXPR struct uniform_t {} uniform;

        /**
         * Indicates that the quantity of message elements may vary in at least one
         * of the processes taking part in an operation.
         * @since 1.0
         */
        MPIWCPP17_CONSTEXPR struct varying_t {} varying;

        /**
         * Indicates that no guarantees on the quantity of message elements are given
         * and a decision must be made dynamically.
         * @since 2.1
         */
        MPIWCPP17_CONSTEXPR struct automatic_t {} automatic;
    }

    inline namespace functor
    {
        /**
         * Indicates that a functor is commutative, enabling MPI to perform possible
         * optimizations when executing specific operations.
         * @since 2.1
         */
        MPIWCPP17_CONSTEXPR struct commutative_t {} commutative;
    }
}

MPIWCPP17_END_NAMESPACE
