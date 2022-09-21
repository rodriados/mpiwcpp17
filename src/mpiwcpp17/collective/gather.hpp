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

MPIWCPP17_BEGIN_NAMESPACE

namespace collective
{
    /**
     * Gathers messages into a single process connected to the communicator, with
     * the behaviour flag of a uniform quantity of elements by each process.
     * @tparam T The message's contents or container type.
     * @param in The message payload to be sent to the root process.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    inline typename payload<T>::return_type gather(
        const payload<T>& in
      , process::rank root = process::root
      , const communicator& comm = world
      , flag::payload::uniform = {}
    ) {
        auto out = (root != comm.rank)
            ? typename payload<T>::return_type ()
            : payload<T>::create(in.count * comm.size);
        guard(MPI_Gather(in, in.count, in.type, out, in.count, in.type, root, comm));
        return out;
    }

    /**
     * Gathers messages into a single processes using lists for defining each process's
     * quantity and displacement of elements.
     * @tparam T The message's contents or container type.
     * @param in The message payload to be sent to the root process.
     * @param count The amount of message elements by each process.
     * @param displ The displacement of each process mesage.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    inline typename payload<T>::return_type gather(
        const payload<T>& in
      , const payload<int>& count
      , const payload<int>& displ
      , process::rank root = process::root
      , const communicator& comm = world
      , flag::payload::varying = {}
    ) {
        auto total = std::accumulate(count.begin(), count.end(), 0);
        auto out = (root != comm.rank)
            ? typename payload<T>::return_type ()
            : payload<T>::create(total);
        guard(MPI_Gatherv(in, in.count, in.type, out, count, displ, in.type, root, comm));
        return out;
    }

    /**
     * Gathers messages into a single process using their natural displacements.
     * @param in The message payload to be sent to the root process.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    inline typename payload<T>::return_type gather(
        const payload<T>& in
      , process::rank root = process::root
      , const communicator& comm = world
      , flag::payload::varying = {}
    ) {
        auto [uniform, count, displ] = detail::calculate_natural_displacements(in.count, comm);
        return uniform
            ? collective::gather<T>(in, root, comm, flag::payload::uniform())
            : collective::gather<T>(in, count, displ, root, comm, flag::payload::varying());
    }

    /**
     * Gathers generic messages into a single process using lists for defining each
     * process's quantity and displacement of elements.
     * @tparam T The message's contents type.
     * @param data The message to be sent to the root process.
     * @param count The amount of message elements by each process.
     * @param displacement The displacement of each process mesage.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @param flag The behaviour flag of varying message sizes.
     * @return The resulting gathered message.
     */
    template <typename T>
    inline typename payload<T>::return_type gather(
        T *data
      , const payload<int>& count
      , const payload<int>& displacement
      , process::rank root = process::root
      , const communicator& comm = world
      , flag::payload::varying flag = {}
    ) {
        auto msg = payload(data, count[comm.rank]);
        return collective::gather<T>(msg, count, displacement, root, comm, flag);
    }

    /**
     * Gathers containers into a single process using lists for defining each process's
     * quantity and displacement of elements.
     * @tparam T The type of container to be gathered.
     * @param data The container to be sent to the root process.
     * @param count The amount of message elements by each process.
     * @param displacement The displacement of each process mesage.
     * @param count The amount of message elements by each process.
     * @param comm The communicator this operation applies to.
     * @param flag The behaviour flag of varying message sizes.
     * @return The resulting gathered message.
     */
    template <typename T>
    inline typename payload<T>::return_type gather(
        T& data
      , const payload<int>& count
      , const payload<int>& displacement
      , process::rank root = process::root
      , const communicator& comm = world
      , flag::payload::varying flag = {}
    ) {
        auto msg = payload(data); msg.count = count[comm.rank];
        return collective::gather<T>(msg, count, displacement, root, comm, flag);
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
    inline typename payload<T>::return_type gather(
        T *data
      , size_t count
      , process::rank root = process::root
      , const communicator& comm = world
      , G flag = {}
    ) {
        auto msg = payload(data, count);
        return collective::gather<T>(msg, root, comm, flag);
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
    inline typename payload<T>::return_type gather(
        T& data
      , process::rank root = process::root
      , const communicator& comm = world
      , G flag = {}
    ) {
        auto msg = payload(data);
        return collective::gather<T>(msg, root, comm, flag);
    }

    /**
     * Gathers generic messages into a single process within the world communicator.
     * @tparam T The message's contents type.
     * @tparam G The behaviour flag type.
     * @param data The message to be sent to the root process.
     * @param count The number of elements to be sent by each process.
     * @param root The operation's root process.
     * @param flag The behaviour flag instance.
     * @return The resulting gathered message.
     */
    template <typename T, typename G>
    inline typename payload<T>::return_type gather(T *data, size_t count, process::rank root, G flag)
    {
        auto msg = payload(data, count);
        return collective::gather<T>(msg, root, world, flag);
    }

    /**
     * Gathers containers into a single process within the world communicator.
     * @tparam T The type of container to be gathered.
     * @tparam G The behaviour flag type.
     * @param data The container to be sent to the root process.
     * @param root The operation's root process.
     * @param flag The behaviour flag instance.
     * @return The resulting gathered message.
     */
    template <typename T, typename G>
    inline typename payload<T>::return_type gather(T& data, process::rank root, G flag)
    {
        auto msg = payload(data);
        return collective::gather<T>(msg, root, world, flag);
    }
}

/*
 * Exposing the above-defined collective operation into the project's root namespace,
 * allowing it be called with decreased verbosity.
 */
using collective::gather;

MPIWCPP17_END_NAMESPACE
