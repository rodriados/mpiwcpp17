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
#include <mpiwcpp17/payload.hpp>
#include <mpiwcpp17/guard.hpp>
#include <mpiwcpp17/world.hpp>

#include <mpiwcpp17/detail/collective.hpp>
#include <mpiwcpp17/detail/wrapper.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail::collective
{
    /**
     * Reduces messages from and to all processes using an operator.
     * @tparam F The operator functor's implementation type.
     * @tparam T The message's contents or container type.
     * @param msg The message payload to be reduced across processes.
     * @param lambda The operator functor to reduce messages with.
     * @param comm The communicator this operation applies to.
     * @return The resulting reduced message.
     */
    template <typename F, typename T>
    inline payload_t<T> allreduce(
        const detail::wrapper_t<T>& msg
      , const F& lambda = {}
      , const communicator_t& comm = world
    ) {
        auto out = payload::create<T>(msg.count);
        auto f = detail::collective::resolve_functor<T>(lambda);
        guard(MPI_Allreduce(msg, out, msg.count, msg.type, f, comm));
        return out;
    }
}

namespace collective
{
    /**
     * Reduces generic messages from and to all processes using an operator.
     * @tparam F The operator functor's implementation type.
     * @tparam T The message's contents type.
     * @param data The message to be reduced into all processes.
     * @param count The number of elements to be reduced.
     * @param lambda The operator functor to reduce messages with.
     * @param comm The communicator this operation applies to.
     * @return The resulting reduced message.
     */
    template <typename F, typename T>
    inline auto allreduce(
        T *data
      , const size_t count
      , const F& lambda = {}
      , const communicator_t& comm = world
    ) {
        auto msg = detail::wrapper_t(data, count);
        return detail::collective::allreduce<F>(msg, lambda, comm);
    }

    /**
     * Reduces containers from and to all processes using an operator.
     * @tparam F The operator functor's implementation type.
     * @tparam T The type of container to be reduced.
     * @param data The container to be reduced into all processes.
     * @param lambda The operator functor to reduce messages with.
     * @param comm The communicator this operation applies to.
     * @return The resulting reduced message.
     */
    template <typename F, typename T>
    inline auto allreduce(
        T& data
      , const F& lambda = {}
      , const communicator_t& comm = world
    ) {
        auto msg = detail::wrapper_t(data);
        return detail::collective::allreduce<F>(msg, lambda, comm);
    }
}

/*
 * Exposing the above-defined collective operation into the project's root namespace,
 * allowing it be called with decreased verbosity.
 */
using collective::allreduce;

MPIWCPP17_END_NAMESPACE
