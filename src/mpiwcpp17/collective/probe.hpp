/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI probe collective operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/status.hpp>
#include <mpiwcpp17/global.hpp>
#include <mpiwcpp17/guard.hpp>
#include <mpiwcpp17/tag.hpp>

MPIWCPP17_BEGIN_NAMESPACE

inline namespace collective
{
    /**
     * Inspects an incoming message and retrieves its status.
     * @param source The process to receive the message from.
     * @param tag The message's identifying tag.
     * @param comm The communicator this operation applies to.
     * @return The inspected message status.
     */
    MPIWCPP17_INLINE status_t probe(
        process_t source = process::any
      , tag_t tag = mpiwcpp17::tag::any
      , const communicator_t& comm = world
    ) {
        return MPIWCPP17_GUARD_CALL(status_t, MPI_Probe(source, tag, comm, &_));
    }
}

MPIWCPP17_END_NAMESPACE
