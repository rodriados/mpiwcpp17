/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI send collective operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/global.hpp>
#include <mpiwcpp17/guard.hpp>
#include <mpiwcpp17/tag.hpp>

#include <mpiwcpp17/detail/payload.hpp>

MPIWCPP17_BEGIN_NAMESPACE

inline namespace collective
{
    /**
     * Sends a message to a process connected to a communicator.
     * @tparam T The message's contents or container type.
     * @param msg The message payload to send to a process.
     * @param destiny The process to send the message to.
     * @param tagg The message's identifying tag.
     * @param comm The communicator this operation applies to.
     */
    template <typename T>
    inline void send(
        const detail::payload<T>& msg
      , process::rank destiny = process::root
      , tag::id tagg = tag::any
      , const communicator& comm = world
    ) {
        tag::id tagid = tagg >= 0 ? tagg : MPI_TAG_UB;
        guard(MPI_Send(msg, msg.count, msg.type, destiny, tagid, comm));
    }

    /**
     * Sends a generic message to a process connected to a communicator.
     * @tparam T The message's contents type.
     * @param data The message payload to send to a process.
     * @param count The number of elements in message to send.
     * @param destiny The process to send the message to.
     * @param tagg The message's identifying tag.
     * @param comm The communicator this operation applies to.
     */
    template <typename T>
    inline void send(
        T *data
      , size_t count
      , process::rank destiny = process::root
      , tag::id tagg = tag::any
      , const communicator& comm = world
    ) {
        auto payload = detail::payload(data, count);
        collective::send<T>(payload, destiny, tagg, comm);
    }

    /**
     * Sends a container to a process connected to a communicator.
     * @tparam T The type of container to send.
     * @param data The message container to send to a process.
     * @param destiny The process to send the message to.
     * @param tagg The message's identifying tag.
     * @param comm The communicator this operation applies to.
     */
    template <typename T>
    inline void send(
        T& data
      , process::rank destiny = process::root
      , tag::id tagg = tag::any
      , const communicator& comm = world
    ) {
        auto payload = detail::payload(data);
        collective::send<T>(payload, destiny, tagg, comm);
    }
}

MPIWCPP17_END_NAMESPACE
