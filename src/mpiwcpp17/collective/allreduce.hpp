/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI all-reduce collective operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/datatype.hpp>
#include <mpiwcpp17/global.hpp>
#include <mpiwcpp17/guard.hpp>

#include <mpiwcpp17/detail/functor.hpp>
#include <mpiwcpp17/detail/payload.hpp>

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
    MPIWCPP17_INLINE detail::payload_out_t<T> allreduce(
        const detail::payload_in_t<T>& msg
      , const F& lambda
      , communicator_t comm = world
    ) {
        auto type = datatype::identify<T>();
        auto out = payload::create_output<T>(msg.count);
        auto f = detail::functor::resolve<T>(lambda);
        guard(MPI_Allreduce(msg.ptr, (T*) out, msg.count, type, f, comm));
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
    MPIWCPP17_INLINE auto allreduce(
        T *data
      , size_t count
      , const F& lambda
      , communicator_t comm = world
    ) {
        auto msg = detail::payload_in_t(data, count);
        return detail::collective::allreduce(msg, lambda, comm);
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
    MPIWCPP17_INLINE auto allreduce(
        T& data
      , const F& lambda
      , communicator_t comm = world
    ) {
        auto msg = detail::payload::to_input(data);
        return detail::collective::allreduce(msg, lambda, comm);
    }
}

/*
 * Exposing the above-defined collective operation into the project's root namespace,
 * allowing it be called with decreased verbosity.
 */
using collective::allreduce;

MPIWCPP17_END_NAMESPACE
