/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI all-reduce collective operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/global.hpp>
#include <mpiwcpp17/guard.hpp>

#include <mpiwcpp17/detail/payload.hpp>
#include <mpiwcpp17/detail/collective.hpp>

MPIWCPP17_BEGIN_NAMESPACE

inline namespace collective
{
    /**
     * Reduces messages from and to all processes using an operator.
     * @tparam T The message's contents or container type.
     * @tparam F The operator functor's implementation type.
     * @param in The message payload to be reduced across processes.
     * @param lambda The operator functor to reduce messages with.
     * @param comm The communicator this operation applies to.
     * @return The resulting reduced message.
     */
    template <typename T, typename F>
    inline typename detail::payload<T>::return_type allreduce(
        const detail::payload<T>& in
      , const F& lambda = {}
      , const communicator& comm = world
    ) {
        using R = typename detail::payload<T>::element_type;
        auto f = detail::collective::resolve_functor<R>(lambda);
        auto out = detail::payload<T>::create(in.count);
        guard(MPI_Allreduce(in, out, in.count, in.type, f, comm));
        return out;
    }

    /**
     * Reduces generic messages from and to all processes using an operator.
     * @tparam T The message's contents type.
     * @tparam F The operator functor's implementation type.
     * @param data The message to be reduced into all processes.
     * @param count The number of elements to be reduced.
     * @param lambda The operator functor to reduce messages with.
     * @param comm The communicator this operation applies to.
     * @return The resulting reduced message.
     */
    template <typename T, typename F>
    inline typename detail::payload<T>::return_type allreduce(
        T *data
      , size_t count
      , const F& lambda = {}
      , const communicator& comm = world
    ) {
        auto payload = detail::payload(data, count);
        return collective::allreduce<T>(payload, lambda, comm);
    }

    /**
     * Reduces containers from and to all processes using an operator.
     * @tparam T The type of container to be reduced.
     * @tparam F The operator functor's implementation type.
     * @param data The container to be reduced into all processes.
     * @param lambda The operator functor to reduce messages with.
     * @param comm The communicator this operation applies to.
     * @return The resulting reduced message.
     */
    template <typename T, typename F>
    inline typename detail::payload<T>::return_type allreduce(
        T& data
      , const F& lambda = {}
      , const communicator& comm = world
    ) {
        auto payload = detail::payload(data);
        return collective::allreduce<T>(payload, lambda, comm);
    }
}

MPIWCPP17_END_NAMESPACE
