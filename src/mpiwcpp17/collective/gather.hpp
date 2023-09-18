/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI gather collective operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <cstdint>
#include <numeric>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/payload.hpp>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/guard.hpp>
#include <mpiwcpp17/world.hpp>
#include <mpiwcpp17/flag.hpp>

#include <mpiwcpp17/collective/allgather.hpp>
#include <mpiwcpp17/detail/wrapper.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail::collective
{
    /**
     * Gathers messages into a single process connected to the communicator, with
     * the behaviour flag of a uniform quantity of elements by each process.
     * @tparam T The message's contents or container type.
     * @param msg The message payload to be sent to the root process.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    inline payload_t<T> gather(
        const detail::wrapper_t<T>& msg
      , const process_t root = process::root
      , const communicator_t& comm = world
      , flag::payload::uniform = {}
    ) {
        auto out = (root == comm.rank)
            ? payload::create<T>(msg.count * comm.size)
            : payload_t<T>();
        guard(MPI_Gather(msg, msg.count, msg.type, out, msg.count, msg.type, root, comm));
        return out;
    }

    /**
     * Gathers messages into a single processes using lists for defining each process's
     * quantity and displacement of elements.
     * @tparam T The message's contents or container type.
     * @param msg The message payload to be sent to the root process.
     * @param count The amount of message elements by each process.
     * @param displ The displacement of each process mesage.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    inline payload_t<T> gather(
        const detail::wrapper_t<T>& msg
      , const payload_t<int>& count
      , const payload_t<int>& displ
      , const process_t root = process::root
      , const communicator_t& comm = world
      , flag::payload::varying = {}
    ) {
        auto out = (root == comm.rank)
            ? payload::create<T>(std::accumulate(std::begin(count), std::end(count), 0))
            : payload_t<T>();
        guard(MPI_Gatherv(msg, msg.count, msg.type, out, count, displ, msg.type, root, comm));
        return out;
    }

    /**
     * Gathers messages into a single process using their natural displacements.
     * @param msg The message payload to be sent to the root process.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    inline payload_t<T> gather(
        const detail::wrapper_t<T>& msg
      , const process_t root = process::root
      , const communicator_t& comm = world
      , flag::payload::varying = {}
    ) {
        auto [uniform, count, displ] = check_uniformity((int)msg.count, comm);
        return uniform
            ? detail::collective::gather(msg, root, comm, flag::payload::uniform())
            : detail::collective::gather(msg, count, displ, root, comm, flag::payload::varying());
    }
}

namespace collective
{
    /**
     * Gathers generic messages into a single process using lists for defining each
     * process's quantity and displacement of elements.
     * @tparam T The message's contents type.
     * @param data The message to be sent to the root process.
     * @param count The amount of message elements by each process.
     * @param displ The displacement of each process mesage.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @param flag The behaviour flag of varying message sizes.
     * @return The resulting gathered message.
     */
    template <typename T>
    inline auto gather(
        T *data
      , const payload_t<int>& count
      , const payload_t<int>& displ
      , const process_t root = process::root
      , const communicator_t& comm = world
      , flag::payload::varying flag = {}
    ) {
        auto msg = detail::wrapper_t(data, count[comm.rank]);
        return detail::collective::gather(msg, count, displ, root, comm, flag);
    }

    /**
     * Gathers containers into a single process using lists for defining each process's
     * quantity and displacement of elements.
     * @tparam T The type of container to be gathered.
     * @param data The container to be sent to the root process.
     * @param count The amount of message elements by each process.
     * @param displ The displacement of each process mesage.
     * @param count The amount of message elements by each process.
     * @param comm The communicator this operation applies to.
     * @param flag The behaviour flag of varying message sizes.
     * @return The resulting gathered message.
     */
    template <typename T>
    inline auto gather(
        T& data
      , const payload_t<int>& count
      , const payload_t<int>& displ
      , const process_t root = process::root
      , const communicator_t& comm = world
      , flag::payload::varying flag = {}
    ) {
        auto msg = detail::wrapper_t(data, count[comm.rank]);
        return detail::collective::gather(msg, count, displ, root, comm, flag);
    }

    /**
     * Gathers generic messages into a single process within communicator.
     * @tparam T The message's contents type.
     * @tparam G The behaviour flag type.
     * @param data The message to be sent to the root process.
     * @param count The number of elements to be sent by each process.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @param flag The behaviour flag instance.
     * @return The resulting gathered message.
     */
    template <typename T, typename G = flag::payload::varying>
    inline auto gather(
        T *data
      , const size_t count
      , const process_t root = process::root
      , const communicator_t& comm = world
      , G flag = {}
    ) {
        auto msg = detail::wrapper_t(data, count);
        return detail::collective::gather(msg, root, comm, flag);
    }

    /**
     * Gathers containers into a single process within communicator.
     * @tparam T The type of container to be gathered.
     * @tparam G The behaviour flag type.
     * @param data The container to be sent to the root process.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @param flag The behaviour flag instance.
     * @return The resulting gathered message.
     */
    template <typename T, typename G = flag::payload::varying>
    inline auto gather(
        T& data
      , const process_t root = process::root
      , const communicator_t& comm = world
      , G flag = {}
    ) {
        auto msg = detail::wrapper_t(data);
        return detail::collective::gather(msg, root, comm, flag);
    }
}

/*
 * Exposing the above-defined collective operation into the project's root namespace,
 * allowing it be called with decreased verbosity.
 */
using collective::gather;

MPIWCPP17_END_NAMESPACE
