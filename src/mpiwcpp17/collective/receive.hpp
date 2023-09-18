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

namespace detail::collective
{
    /**
     * Waits and receives a message from a process connected to a communicator.
     * @tparam T The message's contents or container type.
     * @param count The number of elements in message to be received.
     * @param source The process to receive the message from.
     * @param tag The message's identifying tag.
     * @param comm The communicator this operation applies to.
     * @return The message that has been received.
     */
    template <typename T>
    inline std::tuple<status_t, payload_t<T>> receive(
        const size_t count
      , const process_t source = process::any
      , const tag_t tag = mpiwcpp17::tag::any
      , const communicator_t& comm = world
    ) {
        auto out = payload::create<T>(count);
        status_t s; guard(MPI_Recv(out, out.count, out.type, source, tag, comm, s));
        return std::make_tuple(s, std::move(out));
    }
}

namespace collective
{
    /**
     * Receives a generic message from a process connected to a communicator.
     * @tparam T The message's contents or container type.
     * @param source The process to receive the message from.
     * @param tag The message's identifying tag.
     * @param comm The communicator this operation applies to.
     * @return The message that has been received.
     */
    template <typename T>
    inline std::tuple<status_t, payload_t<T>> receive(
        const process_t source = process::any
      , const tag_t tag = mpiwcpp17::tag::any
      , const communicator_t& comm = world
    ) {
        auto probed = collective::probe(source, tag, comm);
        auto count = status::count<T>(probed);
        return detail::collective::receive<T>(count, source, tag, comm);
    }
}

/*
 * Exposing the above-defined collective operation into the project's root namespace,
 * allowing it be called with decreased verbosity.
 */
using collective::receive;

MPIWCPP17_END_NAMESPACE
