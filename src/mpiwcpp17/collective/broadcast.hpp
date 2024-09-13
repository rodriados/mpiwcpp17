/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI broadcast collective operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>
#include <utility>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/datatype.hpp>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/global.hpp>
#include <mpiwcpp17/guard.hpp>

#include <mpiwcpp17/detail/payload.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail::collective
{
    /**
     * Performs an in-place broadcast operation.
     * @tparam T The message's contents type.
     * @param msg The message to be broadcast in-place to all processes.
     * @param count The total number of elements to be broadcast.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     */
    template <typename T>
    MPIWCPP17_INLINE void broadcast_replace(
        T *msg
      , size_t count
      , process_t root
      , communicator_t comm
    ) {
        auto type = datatype::identify<T>();
        guard(MPI_Bcast(msg, count, type, root, comm));
    }

    /**
     * Broadcasts a message to all processes connected to a communicator.
     * @tparam T The message's contents type.
     * @param msg The message payload to be broadcast to all processes.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @return The message that has been broadcast.
     */
    template <typename T>
    MPIWCPP17_INLINE detail::payload_out_t<T> broadcast(
        const detail::payload_in_t<T>& msg
      , process_t root
      , communicator_t comm
    ) {
        auto out = (root == rank(comm))
            ? payload::copy_to_output(msg)
            : payload::create_output<T>(msg.count);
        broadcast_replace((T*) out, out.count, root, comm);
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
    MPIWCPP17_INLINE auto broadcast(
        T *data
      , size_t count
      , process_t root = process::root
      , communicator_t comm = world
    ) {
        auto msg = detail::payload_in_t(data, count);
        detail::collective::broadcast_replace(&msg.count, 1, root, comm);
        return detail::collective::broadcast(msg, root, comm);
    }

    /**
     * Broadcasts elements to all processes connected to a communicator.
     * @tparam T The type of the elements to be broadcast.
     * @param data The elements to be broadcast to all processes.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @return The message that has been broadcast.
     */
    template <typename T>
    MPIWCPP17_INLINE auto broadcast(
        T& data
      , process_t root = process::root
      , communicator_t comm = world
    ) {
        auto msg = detail::payload::to_input(data);
        if constexpr (detail::payload::is_contiguous_iterable_v<T>)
            detail::collective::broadcast_replace(&msg.count, 1, root, comm);
        return detail::collective::broadcast(msg, root, comm);
    }
}

/*
 * Exposing the above-defined collective operation into the project's root namespace,
 * allowing it be called with decreased verbosity.
 */
using collective::broadcast;

MPIWCPP17_END_NAMESPACE
