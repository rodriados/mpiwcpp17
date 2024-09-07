/**
 * A thin C++17 wrapper for MPI.
 * @file Flags for behaviour guarantees of collective operations.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpiwcpp17/environment.h>

MPIWCPP17_BEGIN_NAMESPACE

namespace flag
{
    MPIWCPP17_INLINE namespace payload
    {
        /**
         * Guarantees that the quantity of message elements is uniform across all
         * processes taking part in a collective operation, either by sending or
         * receiving the same number of elements.
         * @since 1.0
         */
        typedef struct{} uniform_t;

        /**
         * Indicates that the quantity of message elements may vary in at least
         * one of the processes taking part in a collective operation.
         * @since 1.0
         */
        typedef struct{} varying_t;
    }

    MPIWCPP17_INLINE namespace functor
    {
        /**
         * Indicates that a functor is commutative, allowing MPI to perform optimizations
         * when executing specific collective functions.
         * @since 3.0
         */
        typedef struct{} commutative_t;
    }
}

MPIWCPP17_END_NAMESPACE
