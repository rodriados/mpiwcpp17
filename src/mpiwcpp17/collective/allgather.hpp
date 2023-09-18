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

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/payload.hpp>
#include <mpiwcpp17/guard.hpp>
#include <mpiwcpp17/world.hpp>
#include <mpiwcpp17/flag.hpp>

#include <mpiwcpp17/detail/wrapper.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail::collective
{
    /**
     * Gathers messages from and to all processes connected to the communicator,
     * with the flag of a uniform quantity of elements by each process.
     * @tparam T The message's contents or container type.
     * @param msg The message payload to be sent to all processes.
     * @param comm The communicator this operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    inline payload_t<T> allgather(
        const detail::wrapper_t<T>& msg
      , const communicator_t& comm = world
      , flag::payload::uniform = {}
    ) {
        auto out = payload::create<T>(msg.count * comm.size);
        guard(MPI_Allgather(msg, msg.count, msg.type, out, msg.count, msg.type, comm));
        return out;
    }

    /**
     * Gathers messages from and to all processes using lists for defining each
     * process's quantity and displacement of elements.
     * @tparam T The message's contents or container type.
     * @param msg The message payload to be sent to all processes.
     * @param count The amount of message elements by each process.
     * @param displ The displacement of each process mesage.
     * @param comm The communicator this operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    inline payload_t<T> allgather(
        const detail::wrapper_t<T>& msg
      , const payload_t<int>& count
      , const payload_t<int>& displ
      , const communicator_t& comm = world
      , flag::payload::varying = {}
    ) {
        auto out = payload::create<T>(std::accumulate(std::begin(count), std::end(count), 0));
        guard(MPI_Allgatherv(msg, msg.count, msg.type, out, count, displ, msg.type, comm));
        return out;
    }

    /**
     * Checks whether the payloads to be gathered are uniform in size, and also
     * calculates their natural displacement, should the payloads not be uniform.
     * @param size The amount of elements to be sent by the current process.
     * @param comm The communicator the operation applies to.
     * @return A tuple of payload's elements quantity and displacement.
     */
    inline auto check_uniformity(int size, const communicator_t& comm)
    {
        auto displ = payload::create<int>(comm.size);
        auto count = detail::collective::allgather<int>({&size}, comm, flag::payload::uniform());
        bool uniform = true;

        for (int32_t j = 0; j < comm.size; ++j) {
            uniform = uniform && (count[0] == count[j]);
            displ[j] = !j ? 0 : (displ[j-1] + count[j-1]);
        }

        return std::make_tuple(uniform, std::move(count), std::move(displ));
    }

    /**
     * Gathers messages from and to all processes using their natural displacements.
     * @param msg The message payload to be sent to all processes.
     * @param comm The communicator this operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    inline payload_t<T> allgather(
        const detail::wrapper_t<T>& msg
      , const communicator_t& comm = world
      , flag::payload::varying = {}
    ) {
        auto [uniform, count, displ] = check_uniformity((int)msg.count, comm);
        return uniform
            ? detail::collective::allgather(msg, comm, flag::payload::uniform())
            : detail::collective::allgather(msg, count, displ, comm, flag::payload::varying());
    }
}

namespace collective
{
    /**
     * Gathers generic messages from and to all processes using lists for defining
     * each process's quantity and displacement of elements.
     * @tparam T The message's contents type.
     * @param data The message to be sent to all processes.
     * @param count The amount of message elements by each process.
     * @param displ The displacement of each process mesage.
     * @param comm The communicator this operation applies to.
     * @param flag The behaviour flag of varying message sizes.
     * @return The resulting gathered message.
     */
    template <typename T>
    inline auto allgather(
        T *data
      , const payload_t<int>& count
      , const payload_t<int>& displ
      , const communicator_t& comm = world
      , flag::payload::varying flag = {}
    ) {
        auto msg = detail::wrapper_t(data, count[comm.rank]);
        return detail::collective::allgather(msg, count, displ, comm, flag);
    }

    /**
     * Gathers containers from and to all processes using lists for defining each
     * process's quantity and displacement of elements.
     * @tparam T The type of container to be gathered.
     * @param data The container to be sent to all processes.
     * @param count The amount of message elements by each process.
     * @param displ The displacement of each process mesage.
     * @param comm The communicator this operation applies to.
     * @param flag The behaviour flag of varying message sizes.
     * @return The resulting gathered message.
     */
    template <typename T>
    inline auto allgather(
        T& data
      , const payload_t<int>& count
      , const payload_t<int>& displ
      , const communicator_t& comm = world
      , flag::payload::varying flag = {}
    ) {
        auto msg = detail::wrapper_t(data, count[comm.rank]);
        return detail::collective::allgather(msg, count, displ, comm, flag);
    }

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
    template <typename T, typename G = flag::payload::varying>
    inline auto allgather(
        T *data
      , const size_t count
      , const communicator_t& comm = world
      , G flag = {}
    ) {
        auto msg = detail::wrapper_t(data, count);
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
    template <typename T, typename G = flag::payload::varying>
    inline auto allgather(
        T& data
      , const communicator_t& comm = world
      , G flag = {}
    ) {
        auto msg = detail::wrapper_t(data);
        return detail::collective::allgather(msg, comm, flag);
    }
}

/*
 * Exposing the above-defined collective operation into the project's root namespace,
 * allowing it be called with decreased verbosity.
 */
using collective::allgather;

MPIWCPP17_END_NAMESPACE
