/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI barrier collective operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/global.hpp>
#include <mpiwcpp17/guard.hpp>

MPIWCPP17_BEGIN_NAMESPACE

inline namespace collective
{
    /**
     * Blocks execution until all processes within the given communicator have reached
     * the barrier and are, thus, synchronized.
     * @param comm The communicator on to which operation must be performed.
     */
    inline void barrier(const communicator& comm = world)
    {
        guard(MPI_Barrier(comm));
    }
}

MPIWCPP17_END_NAMESPACE
