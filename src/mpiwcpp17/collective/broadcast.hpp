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
#include <mpiwcpp17/global.hpp>
#include <mpiwcpp17/guard.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace collective
{
    /**
     * Broadcasts a message to all processes connected to a communicator.
     * @tparam T The message's contents or container type.
     * @param in The message payload to be broadcast to all processes.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @return The message that has been broadcast.
     */
    template <typename T>
    inline typename payload<T>::return_type broadcast(
        const payload<T>& in
      , process::rank root = process::root
      , const communicator& comm = world
    ) {
        auto out = (root == comm.rank)
            ? static_cast<typename payload<T>::return_type>(in)
            : payload<T>::create(in.count);
        guard(MPI_Bcast(out, out.count, out.type, root, comm));
        return out;
    }

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
    inline typename payload<T>::return_type broadcast(
        T *data
      , size_t count
      , process::rank root = process::root
      , const communicator& comm = world
    ) {
        auto msg = payload(data, count);
        return collective::broadcast<T>(msg, root, comm);
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
    inline typename payload<T>::return_type broadcast(
        T& data
      , process::rank root = process::root
      , const communicator& comm = world
    ) {
        auto msg = payload(data);
        msg.count = collective::broadcast<size_t>({&msg.count, 1}, root, comm);
        return collective::broadcast<T>(msg, root, comm);
    }
}

/*
 * Exposing the above-defined collective operation into the project's root namespace,
 * allowing it be called with decreased verbosity.
 */
using collective::broadcast;

MPIWCPP17_END_NAMESPACE
