/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI scatter collective operation.
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
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/guard.hpp>
#include <mpiwcpp17/world.hpp>
#include <mpiwcpp17/flag.hpp>

#include <mpiwcpp17/collective/broadcast.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace collective
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
    inline typename payload_t<T>::return_t scatter(
        const payload_t<T>& in
      , const process_t root = process::root
      , const communicator_t& comm = world
      , flag::payload::uniform = {}
    ) {
        auto out = payload::create<T>(in.count / comm.size);
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
    inline typename payload_t<T>::return_t scatter(
        const payload_t<T>& in
      , const payload_t<int>& count
      , const payload_t<int>& displ
      , const process_t root = process::root
      , const communicator_t& comm = world
      , flag::payload::varying = {}
    ) {
        auto out = payload::create<T>(count[comm.rank]);
        guard(MPI_Scatterv(in, count, displ, in.type, out, out.count, in.type, root, comm));
        return out;
    }

    namespace detail
    {
        /**
         * Calculates the natural distribution of elements to be scattered.
         * @param elements The amount of elements to be sent by root process.
         * @param processes The amount of processes the operation applies to.
         * @return A tuple of payload size and displacement by process.
         */
        inline auto calculate_distribution(size_t elements, size_t processes)
        {
            payload_t<int> count, displ;

            int32_t quotient  = elements / processes;
            int32_t remainder = elements % processes;
            bool uniform = (remainder == 0);

            if (!uniform) {
                count = payload::create<int>(processes);
                displ = payload::create<int>(processes);

                for (int32_t i = 0; i < processes; ++i) {
                    count[i] = quotient + (remainder > i);
                    displ[i] = (i <= 0) ? 0 : (displ[i-1] + count[i-1]);
                }
            }

            return std::make_tuple(uniform, count, displ);
        }
    }

    /**
     * Scatters a message from the root process using their natural distribution.
     * @tparam T The message's contents or container type.
     * @param in The message payload to be scattered from the root process.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @return The resulting scattered message.
     */
    template <typename T>
    inline typename payload_t<T>::return_t scatter(
        const payload_t<T>& in
      , const process_t root = process::root
      , const communicator_t& comm = world
      , flag::payload::varying = {}
    ) {
        auto [uniform, count, displ] = detail::calculate_distribution(in.count, comm.size);
        return uniform
            ? collective::scatter(in, root, comm, flag::payload::uniform())
            : collective::scatter(in, count, displ, root, comm, flag::payload::varying());
    }

    /**
     * Scatters a generic message from the root process using lists for defining
     * the quantity and displacement of elements for each process.
     * @tparam T The message's contents type.
     * @param data The message to be scattered from the root process.
     * @param count The amount of message elements to each process.
     * @param displ The displacement of each process mesage.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @param flag The behaviour flag of varying message sizes.
     * @return The resulting scattered message.
     */
    template <typename T>
    inline typename payload_t<T>::return_t scatter(
        T *data
      , const payload_t<int>& count
      , const payload_t<int>& displ
      , const process_t root = process::root
      , const communicator_t& comm = world
      , flag::payload::varying flag = {}
    ) {
        auto msg = payload_t(data);
        return collective::scatter<T>(msg, count, displ, root, comm, flag);
    }

    /**
     * Scatters a container from the root process using lists for defining the amount
     * and displacement of elements for each process.
     * @tparam T The type of container to be scattered.
     * @param data The container to be scattered from the root process.
     * @param count The amount of message elements to each process.
     * @param displ The displacement of each process mesage.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @param flag The behaviour flag of varying message sizes.
     * @return The resulting scattered message.
     */
    template <typename T>
    inline typename payload_t<T>::return_t scatter(
        T& data
      , const payload_t<int>& count
      , const payload_t<int>& displ
      , const process_t root = process::root
      , const communicator_t& comm = world
      , flag::payload::varying flag = {}
    ) {
        auto msg = payload_t(data);
        return collective::scatter<T>(msg, count, displ, root, comm, flag);
    }

    /**
     * Scatters a generic message through the processes within communicator.
     * @tparam T The message's contents type.
     * @tparam G The behaviour flag type.
     * @param data The message to be scattered from the root process.
     * @param count The number of elements to be sent to each process.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @param flag The behaviour flag instance.
     * @return The resulting scattered message.
     */
    template <typename T, typename G = flag::payload::varying>
    inline typename payload_t<T>::return_t scatter(
        T *data
      , const size_t count
      , const process_t root = process::root
      , const communicator_t& comm = world
      , G flag = {}
    ) {
        auto msg = payload_t(data, count);
        return collective::scatter<T>(msg, root, comm, flag);
    }

    /**
     * Scatters a container through the processes within communicator.
     * @tparam T The type of container to be scattered.
     * @tparam G The behaviour flag type.
     * @param data The container to be scattered from the root process.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @param flag The behaviour flag instance.
     * @return The resulting scattered message.
     */
    template <typename T, typename G = flag::payload::varying>
    inline typename payload_t<T>::return_t scatter(
        T& data
      , const process_t root = process::root
      , const communicator_t& comm = world
      , G flag = {}
    ) {
        auto msg = payload_t(data);
        msg.count = collective::broadcast<size_t>({&msg.count, 1}, root, comm);
        return collective::scatter<T>(msg, root, comm, flag);
    }

    /**
     * Scatters a generic message through the processes within world communicator.
     * @tparam T The message's contents type.
     * @tparam G The behaviour flag type.
     * @param data The message to be scattered from the root process.
     * @param count The number of elements to be sent to each process.
     * @param root The operation's root process.
     * @param flag The behaviour flag instance.
     * @return The resulting scattered message.
     */
    template <typename T, typename G>
    inline typename payload_t<T>::return_t scatter(
        T *data
      , const size_t count
      , const process_t root
      , G flag
    ) {
        return collective::scatter<T,G>(data, count, root, world, flag);
    }

    /**
     * Scatters a container through the processes within world communicator.
     * @tparam T The type of container to be scattered.
     * @tparam G The behaviour flag type.
     * @param data The container to be scattered from the root process.
     * @param root The operation's root process.
     * @param flag The behaviour flag instance.
     * @return The resulting scattered message.
     */
    template <typename T, typename G>
    inline typename payload_t<T>::return_t scatter(T& data, const process_t root, G flag)
    {
        return collective::scatter<T,G>(data, root, world, flag);
    }
}

/*
 * Exposing the above-defined collective operation into the project's root namespace,
 * allowing it be called with decreased verbosity.
 */
using collective::scatter;

MPIWCPP17_END_NAMESPACE
