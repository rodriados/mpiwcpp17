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
#include <tuple>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/datatype.hpp>
#include <mpiwcpp17/global.hpp>
#include <mpiwcpp17/guard.hpp>
#include <mpiwcpp17/flag.hpp>

#include <mpiwcpp17/detail/payload.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail::collective
{
    /**
     * Gathers messages from and to all processes connected to the communicator,
     * with the flag of a uniform quantity of elements by each process.
     * @tparam T The message's contents type.
     * @param msg The message payload to be sent to all processes.
     * @param comm The communicator this operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    MPIWCPP17_INLINE detail::payload_out_t<T> allgather(
        const detail::payload_in_t<T>& msg
      , communicator_t comm = world
      , flag::payload::uniform_t = {}
    ) {
        auto type = datatype::identify<T>();
        auto out = payload::create_output<T>(msg.count * size(comm));
        guard(MPI_Allgather(msg.ptr, msg.count, type, (T*) out, msg.count, type, comm));
        return out;
    }

    /**
     * Gathers messages from and to all processes using lists for defining each
     * process's quantity and displacement of elements.
     * @tparam T The message's contents type.
     * @param msg The message payload to be sent to all processes.
     * @param total The amount of message elements by each process.
     * @param displ The displacement of each process mesage.
     * @param comm The communicator this operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    MPIWCPP17_INLINE detail::payload_out_t<T> allgather(
        const detail::payload_in_t<T>& msg
      , const detail::payload_in_t<int>& total
      , const detail::payload_in_t<int>& displ
      , communicator_t comm = world
      , flag::payload::varying_t = {}
    ) {
        auto type = datatype::identify<T>();
        auto out = payload::create_output<T>(std::accumulate(total.ptr, total.ptr + total.count, 0));
        guard(MPI_Allgatherv(msg.ptr, msg.count, type, (T*) out, total.ptr, displ.ptr, type, comm));
        return out;
    }

    /**
     * Checks whether the number of payload's elements to be operated is uniform
     * across every process. Also, calculates the natural displacement of each payload.
     * @param count The number of elements to operate in the calling process.
     * @param total The produced total number of elements in each process.
     * @param displ The natural displacement of each process's elements.
     * @return Is the payload element count uniform across all processes?
     */
    MPIWCPP17_INLINE bool check_uniformity(
        int count
      , detail::payload_out_t<int>& total
      , detail::payload_out_t<int>& displ
      , communicator_t comm
    ) {
        bool uniform = true;
        size_t nproc = mpiwcpp17::size(comm);

        displ = payload::create_output<int>(nproc);
        total = allgather<int>({&count, 1}, comm, flag::payload::uniform_t());

        for (size_t j = 0; j < nproc; ++j) {
            uniform = uniform && (total[0] == total[j]);
            displ[j] = !j ? 0 : (displ[j-1] + total[j-1]);
        }

        return uniform;
    }

    /**
     * Gathers messages from and to all processes using their natural displacements.
     * @param msg The message payload to be sent to all processes.
     * @param comm The communicator this operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    MPIWCPP17_INLINE detail::payload_out_t<T> allgather(
        const detail::payload_in_t<T>& msg
      , communicator_t comm = world
      , flag::payload::varying_t = {}
    ) {
        detail::payload_out_t<int> mtotal, mdispl;

        auto mlength = static_cast<int>(msg.count);
        auto uniform = check_uniformity(mlength, mtotal, mdispl, comm);

        const auto displ = payload::to_input(mdispl);
        const auto total = payload::to_input(mtotal);

        return uniform
            ? allgather(msg, comm, flag::payload::uniform_t())
            : allgather(msg, total, displ, comm, flag::payload::varying_t());
    }
}

namespace collective
{
    /**
     * Gathers generic messages from and to all processes within communicator.
     * @tparam T The message's contents type.
     * @tparam G The behaviour flag type.
     * @param data The message to be sent to all processes.
     * @param count The number of elements to be sent by each process.
     * @param comm The communicator this operation applies to.
     * @param flag The behaviour flag instance.
     * @return The resulting gathered message.
     */
    template <typename T, typename G = flag::payload::varying_t>
    MPIWCPP17_INLINE auto allgather(
        T *data
      , size_t count
      , communicator_t comm = world
      , G flag = {}
    ) {
        auto msg = detail::payload_in_t(data, count);
        return detail::collective::allgather(msg, comm, flag);
    }

    /**
     * Gathers containers from and to all processes within communicator.
     * @tparam T The type of container to be gathered.
     * @tparam G The behaviour flag type.
     * @param data The container to be sent to all processes.
     * @param comm The communicator this operation applies to.
     * @param flag The behaviour flag instance.
     * @return The resulting gathered message.
     */
    template <typename T, typename G = flag::payload::varying_t>
    MPIWCPP17_INLINE auto allgather(
        T& data
      , communicator_t comm = world
      , G flag = {}
    ) {
        auto msg = detail::payload::to_input(data);
        return detail::collective::allgather(msg, comm, flag);
    }
}

/*
 * Exposing the above-defined collective operation into the project's root namespace,
 * allowing it be called with decreased verbosity.
 */
using collective::allgather;

MPIWCPP17_END_NAMESPACE
