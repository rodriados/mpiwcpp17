/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI all-gather collective operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <cstdint>
#include <numeric>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/global.hpp>
#include <mpiwcpp17/guard.hpp>

#include <mpiwcpp17/detail/payload.hpp>

MPIWCPP17_BEGIN_NAMESPACE

inline namespace collective
{
    /**
     * Gathers messages from and to all processes connected to the communicator.
     * @tparam T The message's contents or container type.
     * @param in The message payload to be sent to all processes.
     * @param comm The communicator this operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    inline typename detail::payload<T>::return_type allgather(
        const detail::payload<T>& in
      , const communicator& comm = world
    ) {
        auto out = detail::payload<T>::create(in.count * comm.size);
        guard(MPI_Allgather(in, in.count, in.type, out, in.count, in.type, comm));
        return out;
    }

    /**
     * Gathers heterogeneous messages from and to all processes.
     * @tparam T The message's contents or container type.
     * @param in The message payload to be sent to all processes.
     * @param count The amount of message elements by each process.
     * @param displ The displacement of each process mesage.
     * @param comm The communicator this operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    inline typename detail::payload<T>::return_type allgather(
        const detail::payload<T>& in
      , const detail::payload<int>& count
      , const detail::payload<int>& displ
      , const communicator& comm = world
    ) {
        auto total = std::accumulate(count.begin(), count.end(), 0);
        auto out = detail::payload<T>::create(total);
        guard(MPI_Allgatherv(in, in.count, in.type, out, count, displ, in.type, comm));
        return out;
    }

    /**
     * Gathers generic messages from and to all processes within communicator.
     * @tparam T The message's contents type.
     * @param data The message to be sent to all processes.
     * @param count The number of elements to be sent by each process.
     * @param comm The communicator this operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    inline typename detail::payload<T>::return_type allgather(
        T *data
      , size_t count
      , const communicator& comm = world
    ) {
        auto payload = detail::payload(data, count);
        return collective::allgather<T>(payload, comm);
    }

    /**
     * Gathers containers from and to all processes within communicator.
     * @tparam T The type of container to be gathered.
     * @param data The container to be sent to all processes.
     * @param comm The communicator this operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    inline typename detail::payload<T>::return_type allgather(
        T& data
      , const communicator& comm = world
    ) {
        auto payload = detail::payload(data);
        auto count = collective::allgather<int>((int*)&payload.count, 1, comm);
        auto displ = detail::payload<int>::create(comm.size);
        bool homogeneous = true;

        for (int32_t i = 0; i < comm.size; ++i) {
            homogeneous = homogeneous && (count.ptr[0] == count.ptr[i]);
            displ.ptr[i] = (i <= 0) ? 0 : (displ.ptr[i-1] + count.ptr[i-1]);
        }

        return homogeneous
            ? collective::allgather<T>(payload, comm)
            : collective::allgather<T>(payload, count, displ, comm);
    }
}

MPIWCPP17_END_NAMESPACE
