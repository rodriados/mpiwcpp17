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
#include <mpiwcpp17/guard.hpp>
#include <mpiwcpp17/world.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace collective
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

/*
 * Exposing the above-defined collective operation into the project's root namespace,
 * allowing it be called with decreased verbosity.
 */
using collective::barrier;

MPIWCPP17_END_NAMESPACE
