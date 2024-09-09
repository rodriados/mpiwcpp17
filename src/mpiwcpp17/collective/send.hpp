/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI send collective operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/global.hpp>
#include <mpiwcpp17/guard.hpp>
#include <mpiwcpp17/tag.hpp>

#include <mpiwcpp17/detail/payload.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail::collective
{
    /**
     * Sends a message to a process connected to a communicator.
     * @tparam T The message's contents type.
     * @param msg The message payload to send to a process.
     * @param destiny The process to send the message to.
     * @param tag The message's identifying tag.
     * @param comm The communicator this operation applies to.
     */
    template <typename T>
    MPIWCPP17_INLINE void send(
        const detail::payload_in_t<T>& msg
      , process_t destiny = process::root
      , tag_t tag = mpiwcpp17::tag::any
      , communicator_t comm = world
    ) {
        auto type = datatype::identify<T>();
        if (tag < 0) { tag = mpiwcpp17::tag::ub; }
        guard(MPI_Send(msg.ptr, msg.count, type, destiny, tag, comm));
    }
}

namespace collective
{

    /**
     * Sends a generic message to a process connected to a communicator.
     * @tparam T The message's contents type.
     * @param data The message to be sent to a process.
     * @param count The number of elements in message to send.
     * @param destiny The process to send the message to.
     * @param tag The message's identifying tag.
     * @param comm The communicator this operation applies to.
     */
    template <typename T>
    MPIWCPP17_INLINE void send(
        T *data
      , size_t count
      , process_t destiny = process::root
      , tag_t tag = mpiwcpp17::tag::any
      , communicator_t comm = world
    ) {
        auto msg = detail::payload_in_t(data, count);
        detail::collective::send(msg, destiny, tag, comm);
    }

    /**
     * Sends a generic message to a process connected to a communicator.
     * @tparam T The message's contents or container type.
     * @param data The message data to send to a process.
     * @param destiny The process to send the message to.
     * @param tag The message's identifying tag.
     * @param comm The communicator this operation applies to.
     */
    template <typename T>
    MPIWCPP17_INLINE void send(
        T& data
      , process_t destiny = process::root
      , tag_t tag = mpiwcpp17::tag::any
      , communicator_t comm = world
    ) {
        auto msg = detail::payload::to_input(data);
        detail::collective::send(msg, destiny, tag, comm);
    }
}

/*
 * Exposing the above-defined collective operation into the project's root namespace,
 * allowing it be called with decreased verbosity.
 */
using collective::send;

MPIWCPP17_END_NAMESPACE
