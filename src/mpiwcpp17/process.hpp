/**
 * A thin C++17 wrapper for MPI.
 * @file MPI process identifiers and global value definitions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

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
}

MPIWCPP17_END_NAMESPACE
