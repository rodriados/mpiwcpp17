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

MPIWCPP17_BEGIN_NAMESPACE

namespace collective
{
    /**
     * Gathers messages from and to all processes connected to the communicator,
     * with the flag of a uniform quantity of elements by each process.
     * @tparam T The message's contents or container type.
     * @param in The message payload to be sent to all processes.
     * @param comm The communicator this operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    inline typename payload_t<T>::return_t allgather(
        const payload_t<T>& in
      , const communicator_t& comm = world
      , flag::payload::uniform = {}
    ) {
        auto out = payload::create<T>(in.count * comm.size);
        guard(MPI_Allgather(in, in.count, in.type, out, in.count, in.type, comm));
        return out;
    }

    /**
     * Gathers messages from and to all processes using lists for defining each
     * process's quantity and displacement of elements.
     * @tparam T The message's contents or container type.
     * @param in The message payload to be sent to all processes.
     * @param count The amount of message elements by each process.
     * @param displ The displacement of each process mesage.
     * @param comm The communicator this operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    inline typename payload_t<T>::return_t allgather(
        const payload_t<T>& in
      , const payload_t<int>& count
      , const payload_t<int>& displ
      , const communicator_t& comm = world
      , flag::payload::varying = {}
    ) {
        auto total = std::accumulate(count.begin(), count.end(), 0);
        auto out = payload::create<T>(total);
        guard(MPI_Allgatherv(in, in.count, in.type, out, count, displ, in.type, comm));
        return out;
    }

    namespace detail
    {
        /**
         * Calculates the natural displacement of elements to be gathered.
         * @param total The amount of elements to be sent by each process.
         * @param comm The communicator the operation applies to.
         * @return A tuple of payload elements' quantity and displacement.
         */
        inline auto calculate_displacements(size_t total, const communicator_t& comm)
        {
            bool uniform = true;
            auto displacement = payload::create<int>(comm.size);
            auto count = collective::allgather<int>({(int)total}, comm, flag::payload::uniform());

            for (int32_t i = 0; i < comm.size; ++i) {
                uniform = uniform && (count[0] == count[1]);
                displacement[i] = (i <= 0) ? 0 : (displacement[i-1] + count[i-1]);
            }

            return std::make_tuple(uniform, count, displacement);
        }
    }

    /**
     * Gathers messages from and to all processes using their natural displacements.
     * @param in The message payload to be sent to all processes.
     * @param comm The communicator this operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    inline typename payload_t<T>::return_t allgather(
        const payload_t<T>& in
      , const communicator_t& comm = world
      , flag::payload::varying = {}
    ) {
        auto [uniform, count, displ] = detail::calculate_displacements(in.count, comm);
        return uniform
            ? collective::allgather<T>(in, comm, flag::payload::uniform())
            : collective::allgather<T>(in, count, displ, comm, flag::payload::varying());
    }

    /**
     * Gathers generic messages from and to all processes using lists for defining
     * each process's quantity and displacement of elements.
     * @tparam T The message's contents type.
     * @param data The message to be sent to all processes.
     * @param count The amount of message elements by each process.
     * @param displacement The displacement of each process mesage.
     * @param comm The communicator this operation applies to.
     * @param flag The behaviour flag of varying message sizes.
     * @return The resulting gathered message.
     */
    template <typename T>
    inline typename payload_t<T>::return_t allgather(
        T *data
      , const payload_t<int>& count
      , const payload_t<int>& displacement
      , const communicator_t& comm = world
      , flag::payload::varying flag = {}
    ) {
        auto msg = payload_t(data, count[comm.rank]);
        return collective::allgather<T>(msg, count, displacement, comm, flag);
    }

    /**
     * Gathers containers from and to all processes using lists for defining each
     * process's quantity and displacement of elements.
     * @tparam T The type of container to be gathered.
     * @param data The container to be sent to all processes.
     * @param count The amount of message elements by each process.
     * @param displacement The displacement of each process mesage.
     * @param comm The communicator this operation applies to.
     * @param flag The behaviour flag of varying message sizes.
     * @return The resulting gathered message.
     */
    template <typename T>
    inline typename payload_t<T>::return_t allgather(
        T& data
      , const payload_t<int>& count
      , const payload_t<int>& displacement
      , const communicator_t& comm = world
      , flag::payload::varying flag = {}
    ) {
        auto msg = payload_t(data); msg.count = count[comm.rank];
        return collective::allgather<T>(msg, count, displacement, comm, flag);
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
    inline typename payload_t<T>::return_t allgather(
        T *data
      , const size_t count
      , const communicator_t& comm = world
      , G flag = {}
    ) {
        auto msg = payload_t(data, count);
        return collective::allgather<T>(msg, comm, flag);
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
    inline typename payload_t<T>::return_t allgather(
        T& data
      , const communicator_t& comm = world
      , G flag = {}
    ) {
        auto msg = payload_t(data);
        return collective::allgather<T>(msg, comm, flag);
    }

    /**
     * Gathers generic messages from and to all processes within the world communicator.
     * @tparam T The message's contents type.
     * @tparam G The behaviour flag type.
     * @param data The message to be sent to all processes.
     * @param count The number of elements to be sent by each process.
     * @param flag The behaviour flag instance.
     * @return The resulting gathered message.
     */
    template <typename T, typename G>
    inline typename payload_t<T>::return_t allgather(T *data, const size_t count, G flag)
    {
        auto msg = payload_t(data, count);
        return collective::allgather<T>(msg, world, flag);
    }

    /**
     * Gathers containers from and to all processes within the world communicator.
     * @tparam T The type of container to be gathered.
     * @tparam G The behaviour flag type.
     * @param data The container to be sent to all processes.
     * @param flag The behaviour flag instance.
     * @return The resulting gathered message.
     */
    template <typename T, typename G>
    inline typename payload_t<T>::return_t allgather(T& data, G flag)
    {
        auto msg = payload_t(data);
        return collective::allgather<T>(msg, world, flag);
    }
}

/*
 * Exposing the above-defined collective operation into the project's root namespace,
 * allowing it be called with decreased verbosity.
 */
using collective::allgather;

MPIWCPP17_END_NAMESPACE
