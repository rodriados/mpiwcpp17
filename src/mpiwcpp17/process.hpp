/**
 * A thin C++17 wrapper for MPI.
 * @file MPI process identifiers and global value definitions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>
#include <utility>

#include <mpiwcpp17/environment.hpp>

MPIWCPP17_BEGIN_NAMESPACE

/**
 * The type for identifying a specific MPI-process.
 * @since 1.0
 */
using process_t = decltype(MPI_ANY_SOURCE);

namespace process
{
    enum : process_t
    {
        /**
         * The root process identifier within a communicator.
         * @since 1.0
         */
        root = process_t(0)

        /**
         * The special process identifier that may represent any process.
         * @since 1.0
         */
      , any = MPI_ANY_SOURCE

        /**
         * The special process identifier to indicate that an operation must not
         * perform any effect in any process.
         * @since 1.0
         */
      , null = MPI_PROC_NULL
    };

    /**
     * The type of processes within a communicator.
     * @since 2.1
     */
    enum type_t : std::underlying_type<decltype(MPI_COMM_TYPE_SHARED)>::type
    {
        /**
         * This type allows a communicator to be split into subcommunicators, each
         * of which can create a shared memory region.
         * @since 2.1
         */
        shared_memory = MPI_COMM_TYPE_SHARED
      #ifdef OPEN_MPI
        , hwthread = OMPI_COMM_TYPE_HWTHREAD
        , core     = OMPI_COMM_TYPE_CORE
        , l1cache  = OMPI_COMM_TYPE_L1CACHE
        , l2cache  = OMPI_COMM_TYPE_L2CACHE
        , l3cache  = OMPI_COMM_TYPE_L3CACHE
        , socket   = OMPI_COMM_TYPE_SOCKET
        , numa     = OMPI_COMM_TYPE_NUMA
        , board    = OMPI_COMM_TYPE_BOARD
        , host     = OMPI_COMM_TYPE_HOST
        , cluster  = OMPI_COMM_TYPE_CLUSTER
      #endif
    };
}

MPIWCPP17_END_NAMESPACE
