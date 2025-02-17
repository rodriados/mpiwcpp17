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
     * The thread support sets or determines what kind of process and local thread
     * parallelism is supported by the current MPI installation.
     * @since 1.0
     */
    enum class thread_t : int32_t
    {
        /**
         * The first level of thread support.
         * Indicates that the MPI application is single threaded.
         * @since 1.0
         */
        single     = MPI_THREAD_SINGLE

        /**
         * The second level of thread support.
         * Indicates that the MPI application may be multithreaded but all MPI calls
         * will be performed solely by the root or master thread only.
         * @since 1.0
         */
      , funneled   = MPI_THREAD_FUNNELED

        /**
         * The third level of thread support.
         * Indicates that the MPI application may be multithreaded and that any
         * thread might issue MPI calls, however different threads will never issue
         * MPI calls simultaneously.
         * @since 1.0
         */
      , serialized = MPI_THREAD_SERIALIZED

        /**
         * The fourth level of thread support.
         * Indicates that the MPI application may be multithreaded and that any
         * thread might issue MPI calls and, possibly, at the same time.
         * @since 1.0
         */
      , multiple   = MPI_THREAD_MULTIPLE
    };
}

MPIWCPP17_END_NAMESPACE
