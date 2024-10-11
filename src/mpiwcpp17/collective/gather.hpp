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

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/datatype.hpp>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/global.hpp>
#include <mpiwcpp17/guard.hpp>
#include <mpiwcpp17/flag.hpp>

#include <mpiwcpp17/collective/allgather.hpp>
#include <mpiwcpp17/detail/payload.hpp>

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
    MPIWCPP17_INLINE detail::payload_out_t<T> gather(
        const detail::payload_in_t<T>& msg
      , process_t root = process::root
      , communicator_t comm = world
      , flag::payload::uniform_t = {}
    ) {
        auto type = datatype::identify<T>();
        auto out = (root == communicator::rank(comm))
            ? payload::create_output<T>(msg.count * communicator::size(comm))
            : detail::payload_out_t<T>();
        guard(MPI_Gather(msg.ptr, msg.count, type, (T*) out, msg.count, type, root, comm));
        return out;
    }

    /**
     * Gathers messages into a single processes using lists for defining each process's
     * quantity and displacement of elements.
     * @tparam T The message's contents or container type.
     * @param msg The message payload to be sent to the root process.
     * @param total The amount of message elements by each process.
     * @param displ The displacement of each process mesage.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    MPIWCPP17_INLINE detail::payload_out_t<T> gather(
        const detail::payload_in_t<T>& msg
      , const detail::payload_in_t<int>& total
      , const detail::payload_in_t<int>& displ
      , process_t root = process::root
      , communicator_t comm = world
      , flag::payload::varying_t = {}
    ) {
        auto type = datatype::identify<T>();
        auto out = (root == communicator::rank(comm))
            ? payload::create_output<T>(std::accumulate(total.ptr, total.ptr + total.count, 0))
            : detail::payload_out_t<T>();
        guard(MPI_Gatherv(msg.ptr, msg.count, type, (T*) out, total.ptr, displ.ptr, type, root, comm));
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
    MPIWCPP17_INLINE detail::payload_out_t<T> gather(
        const detail::payload_in_t<T>& msg
      , process_t root = process::root
      , communicator_t comm = world
      , flag::payload::varying_t = {}
    ) {
        detail::payload_out_t<int> mtotal, mdispl;

        auto mlength = static_cast<int>(msg.count);
        auto uniform = check_uniformity(mlength, mtotal, mdispl, comm);

        const auto displ = payload::to_input(mdispl);
        const auto total = payload::to_input(mtotal);

        return uniform
            ? gather(msg, root, comm, flag::payload::uniform_t())
            : gather(msg, total, displ, root, comm, flag::payload::varying_t());
    }
}

namespace collective
{
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
    template <typename T, typename G = flag::payload::varying_t>
    MPIWCPP17_INLINE auto gather(
        T *data
      , size_t count
      , process_t root = process::root
      , communicator_t comm = world
      , G flag = {}
    ) {
        auto msg = detail::payload_in_t(data, count);
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
    template <typename T, typename G = flag::payload::varying_t>
    MPIWCPP17_INLINE auto gather(
        T& data
      , process_t root = process::root
      , communicator_t comm = world
      , G flag = {}
    ) {
        auto msg = detail::payload::to_input(data);
        return detail::collective::gather(msg, root, comm, flag);
    }
}

/*
 * Exposing the above-defined collective operation into the project's root namespace,
 * allowing it be called with decreased verbosity.
 */
using collective::gather;

MPIWCPP17_END_NAMESPACE
