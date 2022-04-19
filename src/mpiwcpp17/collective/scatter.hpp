/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI scatter collective operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <cstdint>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/global.hpp>
#include <mpiwcpp17/guard.hpp>

#include <mpiwcpp17/detail/payload.hpp>
#include <mpiwcpp17/collective/broadcast.hpp>

MPIWCPP17_BEGIN_NAMESPACE

inline namespace collective
{
    /**
     * Scatters a message through the processes connected to the communicator.
     * @tparam T The message's contents or container type.
     * @param in The message payload to be scattered from the root process.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @return The resulting scattered message.
     */
    template <typename T>
    inline typename detail::payload<T>::return_type scatter(
        const detail::payload<T>& in
      , process::rank root = process::root
      , const communicator& comm = world
    ) {
        auto out = detail::payload<T>::create(in.count / comm.size);
        guard(MPI_Scatter(in, out.count, in.type, out, out.count, in.type, root, comm));
        return out;
    }

    /**
     * Scatters a message unevenly through the processes of the communicator.
     * @tparam T The message's contents or container type.
     * @param in The message payload to be scattered from the root process.
     * @param count The amount of message elements to each process.
     * @param displ The displacement of each process mesage.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @return The resulting scattered message.
     */
    template <typename T>
    inline typename detail::payload<T>::return_type scatter(
        const detail::payload<T>& in
      , const detail::payload<int>& count
      , const detail::payload<int>& displ
      , process::rank root = process::root
      , const communicator& comm = world
    ) {
        auto out = detail::payload<T>::create(count.ptr[comm.rank]);
        guard(MPI_Scatterv(in, count, displ, in.type, out, out.count, in.type, root, comm));
        return out;
    }

    /**
     * Scatters a generic message through the processes within communicator.
     * @tparam T The message's contents type.
     * @param data The message to be scattered from the root process.
     * @param count The number of elements to be sent to each process.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @return The resulting scattered message.
     */
    template <typename T>
    inline typename detail::payload<T>::return_type scatter(
        T *data
      , size_t count
      , process::rank root = process::root
      , const communicator& comm = world
    ) {
        auto payload = detail::payload(data, count);
        return collective::scatter<T>(payload, root, comm);
    }

    /**
     * Scatters a container through the processes within communicator.
     * @tparam T The type of container to be scattered.
     * @param data The container to be scattered from the root process.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @return The resulting scattered message.
     */
    template <typename T>
    inline typename detail::payload<T>::return_type scatter(
        T& data
      , process::rank root = process::root
      , const communicator& comm = world
    ) {
        auto payload = detail::payload(data);
        payload.count = collective::broadcast(&payload.count, 1, root, comm);

        int32_t quotient  = payload.count / comm.size;
        int32_t remainder = payload.count % comm.size;

        if (!remainder) return collective::scatter<T>(payload, root, comm);

        auto count = detail::payload<int>::create(comm.size);
        auto displ = detail::payload<int>::create(comm.size);

        for (int32_t i = 0; i < comm.size; ++i) {
            count.ptr[i] = quotient + (remainder > i);
            displ.ptr[i] = quotient * i + (i < remainder ? i : remainder);
        }

        return collective::scatter<T>(payload, count, displ, root, comm);
    }
}

MPIWCPP17_END_NAMESPACE
