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
#include <mpiwcpp17/global.hpp>
#include <mpiwcpp17/status.hpp>
#include <mpiwcpp17/guard.hpp>
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
     * @param tagg The message's identifying tag.
     * @param comm The communicator this operation applies to.
     * @return The message that has been received.
     */
    template <typename T>
    inline std::tuple<status, typename payload<T>::return_type> receive(
        const payload<T>& out
      , process::rank source = process::any
      , tag::id tagg = tag::any
      , const communicator& comm = world
    ) {
        status stt; guard(MPI_Recv(out, out.count, out.type, source, tagg, comm, stt));
        return std::make_tuple(stt, out);
    }

    /**
     * Receives a generic message from a process connected to a communicator.
     * @tparam T The message's contents or container type.
     * @param source The process to receive the message from.
     * @param tagg The message's identifying tag.
     * @param comm The communicator this operation applies to.
     * @return The message that has been received.
     */
    template <typename T>
    inline std::tuple<status, typename payload<T>::return_type> receive(
        process::rank source = process::any
      , tag::id tagg = tag::any
      , const communicator& comm = world
    ) {
        using E = typename payload<T>::element_type;
        auto probed = collective::probe(source, tagg, comm);
        auto msg = payload<T>::create(status::count<E>(probed));
        return collective::receive<T>(msg, source, tagg, comm);
    }
}

/*
 * Exposing the above-defined collective operation into the project's root namespace,
 * allowing it be called with decreased verbosity.
 */
using collective::receive;

MPIWCPP17_END_NAMESPACE
