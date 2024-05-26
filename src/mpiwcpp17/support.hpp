/**
 * A thin C++17 wrapper for MPI.
 * @file Miscellaneous functions and classes of MPI features.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>
#include <cstdint>

#include <mpiwcpp17/environment.h>

MPIWCPP17_BEGIN_NAMESPACE

namespace support
{
    /**
     * The level of MPI thread support.
     * @since 1.0
     */
    enum class thread_t : int32_t
    {
        single     = MPI_THREAD_SINGLE
      , funneled   = MPI_THREAD_FUNNELED
      , serialized = MPI_THREAD_SERIALIZED
      , multiple   = MPI_THREAD_MULTIPLE
    };
}

MPIWCPP17_END_NAMESPACE
