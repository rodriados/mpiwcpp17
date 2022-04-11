/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI receive collective operation.
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

#include <mpiwcpp17/detail/payload.hpp>
#include <mpiwcpp17/collective/probe.hpp>

MPIWCPP17_BEGIN_NAMESPACE

inline namespace collective
{
    /**
     * Waits and receives a message from a process connected to a communicator.
     * @tparam T The message's contents or container type.
     * @param msg The message payload to receive from a process.
     * @param source The process to receive the message from.
     * @param tagg The message's identifying tag.
     * @param comm The communicator this operation applies to.
     * @return The message that has been received.
     */
    template <typename T>
    inline typename detail::payload<T>::return_type receive(
        const detail::payload<T>& msg
      , process::rank source = process::any
      , tag::id tagg = tag::any
      , const communicator& comm = world
    ) {
        guard(MPI_Recv(msg, msg.count, msg.type, source, tagg, comm, &status::last));
        return msg;
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
    inline typename detail::payload<T>::return_type receive(
        process::rank source = process::any
      , tag::id tagg = tag::any
      , const communicator& comm = world
    ) {
        using E = typename detail::payload<T>::element_type;
        auto probed = collective::probe(source, tagg, comm);
        auto payload = detail::payload<T>::create(status::count<E>(probed));
        return collective::receive<T>(payload, source, tagg, comm);
    }
}

MPIWCPP17_END_NAMESPACE
