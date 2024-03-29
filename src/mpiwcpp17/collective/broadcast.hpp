/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI broadcast collective operation.
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

#include <mpiwcpp17/detail/wrapper.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail::collective
{
    /**
     * Broadcasts a message to all processes connected to a communicator.
     * @tparam T The message's contents or container type.
     * @param msg The message payload to be broadcast to all processes.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @return The message that has been broadcast.
     */
    template <typename T>
    inline payload_t<T> broadcast(
        const detail::wrapper_t<T>& msg
      , const process_t root = process::root
      , const communicator_t& comm = world
    ) {
        auto out = (root == comm.rank)
            ? detail::wrapper_to_payload(msg)
            : payload::create<T>(msg.count);
        guard(MPI_Bcast(out, out.count, out.type, root, comm));
        return out;
    }
}

namespace collective
{
    /**
     * Broadcasts a generic message to all processes connected to a communicator.
     * @tparam T The message's contents type.
     * @param data The message to be broadcast to all processes.
     * @param count The number of elements to be broadcast.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @return The message that has been broadcast.
     */
    template <typename T>
    inline auto broadcast(
        T *data
      , const size_t count
      , const process_t root = process::root
      , const communicator_t& comm = world
    ) {
        auto msg = detail::wrapper_t(data, count);
        return detail::collective::broadcast(msg, root, comm);
    }

    /**
     * Broadcasts a container to all processes connected to a communicator.
     * @tparam T The type of container to be broadcast.
     * @param data The container to be broadcast to all processes.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @return The message that has been broadcast.
     */
    template <typename T>
    inline auto broadcast(
        T& data
      , const process_t root = process::root
      , const communicator_t& comm = world
    ) {
        auto msg = detail::wrapper_t(data);
        msg.count = detail::collective::broadcast<size_t>(msg.count, root, comm);
        return detail::collective::broadcast(msg, root, comm);
    }
}

/*
 * Exposing the above-defined collective operation into the project's root namespace,
 * allowing it be called with decreased verbosity.
 */
using collective::broadcast;

MPIWCPP17_END_NAMESPACE
