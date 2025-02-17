/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI scatter collective operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <cstdint>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/datatype.hpp>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/global.hpp>
#include <mpiwcpp17/guard.hpp>
#include <mpiwcpp17/flag.hpp>

#include <mpiwcpp17/collective/broadcast.hpp>
#include <mpiwcpp17/detail/payload.hpp>

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
    MPIWCPP17_INLINE detail::payload_out_t<T> scatter(
        const detail::payload_in_t<T>& msg
      , process_t root = process::root
      , communicator_t comm = mpiwcpp17::world
      , flag::payload::uniform_t = {}
    ) {
        auto type = datatype::identify<T>();
        auto out = payload::create_output<T>(msg.count / communicator::size(comm));
        guard(MPI_Scatter(msg.ptr, out.count, type, (T*) out, out.count, type, root, comm));
        return out;
    }

    /**
     * Scatters a message unevenly through the processes of the communicator.
     * @tparam T The message's contents or container type.
     * @param msg The message payload to be scattered from the root process.
     * @param total The amount of message elements to each process.
     * @param displ The displacement of each process mesage.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @return The resulting scattered message.
     */
    template <typename T>
    MPIWCPP17_INLINE detail::payload_out_t<T> scatter(
        const detail::payload_in_t<T>& msg
      , const detail::payload_in_t<int>& total
      , const detail::payload_in_t<int>& displ
      , process_t root = process::root
      , communicator_t comm = mpiwcpp17::world
      , flag::payload::varying_t = {}
    ) {
        auto type = datatype::identify<T>();
        auto out = payload::create_output<T>(total.ptr[communicator::rank(comm)]);
        guard(MPI_Scatterv(msg.ptr, total.ptr, displ.ptr, type, (T*) out, out.count, type, root, comm));
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
    MPIWCPP17_INLINE detail::payload_out_t<T> scatter(
        const detail::payload_in_t<T>& msg
      , process_t root = process::root
      , communicator_t comm = mpiwcpp17::world
      , flag::payload::varying_t = {}
    ) {
        size_t nproc = communicator::size(comm);

        size_t quotient  = msg.count / nproc;
        size_t remainder = msg.count % nproc;

        if (!remainder)
            return detail::collective::scatter(msg, root, comm, flag::payload::uniform_t());

        auto mtotal = payload::create_output<int>(nproc);
        auto mdispl = payload::create_output<int>(nproc);

        for (size_t j = 0; j < nproc; ++j) {
            mtotal[j] = quotient + (remainder > j);
            mdispl[j] = !j ? 0 : (mdispl[j-1] + mtotal[j-1]);
        }

        const auto displ = payload::to_input(mdispl);
        const auto total = payload::to_input(mtotal);

        return detail::collective::scatter(msg, total, displ, root, comm, flag::payload::varying_t());
    }
}

namespace collective
{
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
    template <typename T, typename G = flag::payload::varying_t>
    MPIWCPP17_INLINE auto scatter(
        T *data
      , size_t count
      , process_t root = process::root
      , communicator_t comm = world
      , G flag = {}
    ) {
        auto msg = detail::payload_in_t(data, count);
        detail::collective::broadcast_replace(&msg.count, 1, root, comm);
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
    template <typename T, typename G = flag::payload::varying_t>
    MPIWCPP17_INLINE auto scatter(
        T& data
      , process_t root = process::root
      , communicator_t comm = world
      , G flag = {}
    ) {
        auto msg = detail::payload::to_input(data);
        detail::collective::broadcast_replace(&msg.count, 1, root, comm);
        return detail::collective::scatter(msg, root, comm, flag);
    }
}

/*
 * Exposing the above-defined collective operation into the project's root namespace,
 * allowing it be called with decreased verbosity.
 */
using collective::scatter;

MPIWCPP17_END_NAMESPACE
