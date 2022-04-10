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
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/global.hpp>
#include <mpiwcpp17/guard.hpp>

#include <mpiwcpp17/detail/payload.hpp>

MPIWCPP17_BEGIN_NAMESPACE

inline namespace collective
{
    /**
     * Broadcasts a message to all processes connected to a communicator.
     * @tparam T The message's contents or container type.
     * @param payload The message payload to be broadcast to all processes.
     * @param root The operation's root node.
     * @param comm The communicator this operation applies to.
     * @return The message that has been broadcast.
     */
    template <typename T>
    inline typename detail::payload<T>::return_type broadcast(
        const detail::payload<T>& payload
      , process::rank root = process::root
      , const communicator& comm = world
    ) {
        auto msg = (root == comm.rank)
            ? static_cast<typename detail::payload<T>::return_type>(payload)
            : detail::payload<T>::create(payload.count);
        guard(MPI_Bcast(msg, msg.count, msg.type, root, comm));
        return msg;
    }

    /**
     * Broadcasts a generic message to all processes connected to a communicator.
     * @tparam T The message's contents type.
     * @param data The message to be broadcast to all processes.
     * @param count The number of elements to be broadcast.
     * @param root The operation's root node.
     * @param comm The communicator this operation applies to.
     * @return The message that has been broadcast.
     */
    template <typename T>
    inline typename detail::payload<T>::return_type broadcast(
        T *data
      , size_t count
      , process::rank root = process::root
      , const communicator& comm = world
    ) {
        auto payload = detail::payload(data, count);
        return collective::broadcast<T>(payload, root, comm);
    }

    /**
     * Broadcasts a container to all processes connected to a communicator.
     * @tparam T The type of container to be broadcast.
     * @param data The container to be broadcast to all processes.
     * @param root The operation's root node.
     * @param comm The communicator this operation applies to.
     * @return The message that has been broadcast.
     */
    template <typename T>
    inline typename detail::payload<T>::return_type broadcast(
        T& data
      , process::rank root = process::root
      , const communicator& comm = world
    ) {
        auto payload = detail::payload(data);
        payload.count = collective::broadcast(&payload.count, 1, root, comm);
        return collective::broadcast<T>(payload, root, comm);
    }
}

MPIWCPP17_END_NAMESPACE
