/**
 * A thin C++17 wrapper for MPI.
 * @file Flags for behaviour guarantees of collective operations.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <cstdint>

#include <mpiwcpp17/environment.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace flag
{
    inline namespace payload
    {
        /**
         * Guarantees that the quantity of message elements is uniform across all
         * processes taking part in a collective operation, either by sending or
         * receiving the same number of elements.
         * @since 1.0
         */
        typedef struct{} uniform;

        /**
         * Indicates that the quantity of message elements may vary in at least
         * one of the processes taking part in a collective operation.
         * @since 1.0
         */
        typedef struct{} varying;
    }
}

MPIWCPP17_END_NAMESPACE
