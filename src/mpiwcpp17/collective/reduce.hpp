/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI reduce collective operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <utility>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/global.hpp>
#include <mpiwcpp17/guard.hpp>

#include <mpiwcpp17/detail/payload.hpp>
#include <mpiwcpp17/collective/utility.hpp>

MPIWCPP17_BEGIN_NAMESPACE

inline namespace collective
{
    /**
     * Reduces messages into one process using an operator.
     * @tparam T The message's contents or container type.
     * @tparam F The operator functor's implementation type.
     * @param in The message payload to be reduced across processes.
     * @param lambda The operator functor to reduce messages with.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @return The resulting reduced message.
     */
    template <typename T, typename F>
    inline typename detail::payload<T>::return_type reduce(
        const detail::payload<T>& in
      , const F& lambda = {}
      , process::rank root = process::root
      , const communicator& comm = world
    ) {
        using R = typename detail::payload<T>::element_type;
        auto f = collective::utility::resolvef<R>(lambda);
        auto out = (root != comm.rank)
            ? typename detail::payload<T>::return_type ()
            : detail::payload<T>::create(in.count);
        guard(MPI_Reduce(in, out, in.count, in.type, f, root, comm));
        return out;
    }

    /**
     * Reduces generic messages into one process using an operator.
     * @tparam T The message's contents type.
     * @tparam F The operator functor's implementation type.
     * @param data The message to be reduced into the root process.
     * @param count The number of elements to be reduced.
     * @param lambda The operator functor to reduce messages with.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @return The resulting reduced message.
     */
    template <typename T, typename F>
    inline typename detail::payload<T>::return_type reduce(
        T *data
      , size_t count
      , const F& lambda = {}
      , process::rank root = process::root
      , const communicator& comm = world
    ) {
        auto payload = detail::payload(data, count);
        return collective::reduce<T>(payload, lambda, root, comm);
    }

    /**
     * Reduces containers into one process using an operator.
     * @tparam T The type of container to be reduced.
     * @tparam F The operator functor's implementation type.
     * @param data The container to be reduced into the root process.
     * @param lambda The operator functor to reduce messages with.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @return The resulting reduced message.
     */
    template <typename T, typename F>
    inline typename detail::payload<T>::return_type reduce(
        T& data
      , const F& lambda = {}
      , process::rank root = process::root
      , const communicator& comm = world
    ) {
        auto payload = detail::payload(data);
        return collective::reduce<T>(payload, lambda, root, comm);
    }
}

MPIWCPP17_END_NAMESPACE
