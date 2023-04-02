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
#include <mpiwcpp17/payload.hpp>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/guard.hpp>
#include <mpiwcpp17/world.hpp>
#include <mpiwcpp17/tag.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace collective
{
    /**
     * Sends a message to a process connected to a communicator.
     * @tparam T The message's contents or container type.
     * @param in The message payload to send to a process.
     * @param destiny The process to send the message to.
     * @param tag The message's identifying tag.
     * @param comm The communicator this operation applies to.
     */
    template <typename T>
    inline void send(
        const payload<T>& in
      , process_t destiny = process::root
      , tag_t tag = mpiwcpp17::tag::any
      , const communicator& comm = world
    ) {
        if (tag < 0) { tag = mpiwcpp17::tag::ub; }
        guard(MPI_Send(in, in.count, in.type, destiny, tag, comm));
    }

    /**
     * Sends a generic message to a process connected to a communicator.
     * @tparam T The message's contents type.
     * @param data The message payload to send to a process.
     * @param count The number of elements in message to send.
     * @param destiny The process to send the message to.
     * @param tag The message's identifying tag.
     * @param comm The communicator this operation applies to.
     */
    template <typename T>
    inline void send(
        T *data
      , size_t count
      , process_t destiny = process::root
      , tag_t tag = mpiwcpp17::tag::any
      , const communicator& comm = world
    ) {
        auto msg = payload(data, count);
        collective::send<T>(msg, destiny, tag, comm);
    }

    /**
     * Sends a container to a process connected to a communicator.
     * @tparam T The type of container to send.
     * @param data The message container to send to a process.
     * @param destiny The process to send the message to.
     * @param tag The message's identifying tag.
     * @param comm The communicator this operation applies to.
     */
    template <typename T>
    inline void send(
        T& data
      , process_t destiny = process::root
      , tag_t tag = mpiwcpp17::tag::any
      , const communicator& comm = world
    ) {
        auto msg = payload(data);
        collective::send<T>(msg, destiny, tag, comm);
    }
}

/*
 * Exposing the above-defined collective operation into the project's root namespace,
 * allowing it be called with decreased verbosity.
 */
using collective::send;

MPIWCPP17_END_NAMESPACE
