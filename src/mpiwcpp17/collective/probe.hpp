/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI probe collective operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/global.hpp>
#include <mpiwcpp17/status.hpp>
#include <mpiwcpp17/guard.hpp>
#include <mpiwcpp17/tag.hpp>

MPIWCPP17_BEGIN_NAMESPACE

inline namespace collective
{
    /**
     * Inspects an incoming message and retrieves its status.
     * @param source The process to receive the message from.
     * @param tagg The message's identifying tag.
     * @param comm The communicator this operation applies to.
     * @return The inspected message status.
     */
    inline status probe(
        process::rank source = process::any
      , tag::id tagg = tag::any
      , const communicator& comm = world
    ) {
        status stt; guard(MPI_Probe(source, tagg, comm, stt));
        return stt;
    }
}

MPIWCPP17_END_NAMESPACE
