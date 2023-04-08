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
#include <mpiwcpp17/status.hpp>
#include <mpiwcpp17/guard.hpp>
#include <mpiwcpp17/world.hpp>
#include <mpiwcpp17/tag.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace collective
{
    /**
     * Inspects an incoming message and retrieves its status.
     * @param source The process to receive the message from.
     * @param tag The message's identifying tag.
     * @param comm The communicator this operation applies to.
     * @return The inspected message status.
     */
    inline status_t probe(
        const process_t source = process::any
      , const tag_t tag = mpiwcpp17::tag::any
      , const communicator_t& comm = world
    ) {
        status_t s; guard(MPI_Probe(source, tag, comm, s));
        return s;
    }
}

/*
 * Exposing the above-defined collective operation into the project's root namespace,
 * allowing it be called with decreased verbosity.
 */
using collective::probe;

MPIWCPP17_END_NAMESPACE
