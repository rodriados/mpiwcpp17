/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI receive collective operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <tuple>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/payload.hpp>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/status.hpp>
#include <mpiwcpp17/guard.hpp>
#include <mpiwcpp17/world.hpp>
#include <mpiwcpp17/tag.hpp>

#include <mpiwcpp17/collective/probe.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace collective
{
    /**
     * Waits and receives a message from a process connected to a communicator.
     * @tparam T The message's contents or container type.
     * @param out The message payload to receive from a process.
     * @param source The process to receive the message from.
     * @param tag The message's identifying tag.
     * @param comm The communicator this operation applies to.
     * @return The message that has been received.
     */
    template <typename T>
    inline typename payload_t<T>::return_t receive(
        const payload_t<T>& out
      , const process_t source = process::any
      , const tag_t tag = mpiwcpp17::tag::any
      , const communicator_t& comm = world
    ) {
        guard(MPI_Recv(out, out.count, out.type, source, tag, comm, status::ignore));
        return out;
    }

    /**
     * Receives a generic message from a process connected to a communicator.
     * @tparam T The message's contents or container type.
     * @param source The process to receive the message from.
     * @param tag The message's identifying tag.
     * @param comm The communicator this operation applies to.
     * @return The message that has been received.
     */
    template <typename T>
    inline typename payload_t<T>::return_t receive(
        const process_t source = process::any
      , const tag_t tag = mpiwcpp17::tag::any
      , const communicator_t& comm = world
    ) {
        using E = typename payload_t<T>::element_t;
        auto probed = collective::probe(source, tag, comm);
        auto msg = payload::create<T>(status::count<E>(probed));
        return collective::receive<T>(msg, source, tag, comm);
    }
}

/*
 * Exposing the above-defined collective operation into the project's root namespace,
 * allowing it be called with decreased verbosity.
 */
using collective::receive;

MPIWCPP17_END_NAMESPACE
