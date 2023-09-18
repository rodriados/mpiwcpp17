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
#include <mpiwcpp17/detail/wrapper.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail::collective
{
    /**
     * Scatters a message through the processes connected to the communicator.
     * @tparam T The message's contents or container type.
     * @param msg The message payload to be scattered from the root process.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @return The resulting scattered message.
     */
    template <typename T>
    inline payload_t<T> scatter(
        const detail::wrapper_t<T>& msg
      , const process_t root = process::root
      , const communicator_t& comm = world
      , flag::payload::uniform = {}
    ) {
        auto out = payload::create<T>(msg.count / comm.size);
        guard(MPI_Scatter(msg, out.count, msg.type, out, out.count, msg.type, root, comm));
        return out;
    }

    /**
     * Scatters a message unevenly through the processes of the communicator.
     * @tparam T The message's contents or container type.
     * @param msg The message payload to be scattered from the root process.
     * @param count The amount of message elements to each process.
     * @param displ The displacement of each process mesage.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @return The resulting scattered message.
     */
    template <typename T>
    inline payload_t<T> scatter(
        const detail::wrapper_t<T>& msg
      , const payload_t<int>& count
      , const payload_t<int>& displ
      , const process_t root = process::root
      , const communicator_t& comm = world
      , flag::payload::varying = {}
    ) {
        auto out = payload::create<T>(count[comm.rank]);
        guard(MPI_Scatterv(msg, count, displ, msg.type, out, out.count, msg.type, root, comm));
        return out;
    }

    /**
     * Scatters a message from the root process using their natural distribution.
     * @tparam T The message's contents or container type.
     * @param msg The message payload to be scattered from the root process.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @return The resulting scattered message.
     */
    template <typename T>
    inline payload_t<T> scatter(
        const detail::wrapper_t<T>& msg
      , const process_t root = process::root
      , const communicator_t& comm = world
      , flag::payload::varying = {}
    ) {
        int32_t quotient  = msg.count / comm.size;
        int32_t remainder = msg.count % comm.size;

        if (!remainder)
            return detail::collective::scatter(msg, root, comm, flag::payload::uniform());

        auto count = payload::create<int>(comm.size);
        auto displ = payload::create<int>(comm.size);

        for (int32_t j = 0; j < comm.size; ++j) {
            count[j] = quotient + (remainder > j);
            displ[j] = !j ? 0 : (displ[j-1] + count[j-1]);
        }

        return detail::collective::scatter(msg, count, displ, root, comm, flag::payload::varying());
    }
}

namespace collective
{
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
    inline auto scatter(
        T *data
      , const payload_t<int>& count
      , const payload_t<int>& displ
      , const process_t root = process::root
      , const communicator_t& comm = world
      , flag::payload::varying flag = {}
    ) {
        auto msg = detail::wrapper_t(data);
        return detail::collective::scatter(msg, count, displ, root, comm, flag);
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
    inline auto scatter(
        T& data
      , const payload_t<int>& count
      , const payload_t<int>& displ
      , const process_t root = process::root
      , const communicator_t& comm = world
      , flag::payload::varying flag = {}
    ) {
        auto msg = detail::wrapper_t(data);
        return detail::collective::scatter(msg, count, displ, root, comm, flag);
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
    inline auto scatter(
        T *data
      , const size_t count
      , const process_t root = process::root
      , const communicator_t& comm = world
      , G flag = {}
    ) {
        auto msg = detail::wrapper_t(data, count);
        return detail::collective::scatter(msg, root, comm, flag);
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
    inline auto scatter(
        T& data
      , const process_t root = process::root
      , const communicator_t& comm = world
      , G flag = {}
    ) {
        auto msg = detail::wrapper_t(data);
        msg.count = detail::collective::broadcast<size_t>(msg.count, root, comm);
        return detail::collective::scatter(msg, root, comm, flag);
    }
}

/*
 * Exposing the above-defined collective operation into the project's root namespace,
 * allowing it be called with decreased verbosity.
 */
using collective::scatter;

MPIWCPP17_END_NAMESPACE
