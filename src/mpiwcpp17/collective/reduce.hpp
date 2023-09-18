/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI reduce collective operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/payload.hpp>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/guard.hpp>
#include <mpiwcpp17/world.hpp>

#include <mpiwcpp17/detail/collective.hpp>
#include <mpiwcpp17/detail/wrapper.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail::collective
{
    /**
     * Reduces messages into one process using an operator.
     * @tparam F The operator functor's implementation type.
     * @tparam T The message's contents or container type.
     * @param msg The message payload to be reduced across processes.
     * @param lambda The operator functor to reduce messages with.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @return The resulting reduced message.
     */
    template <typename F, typename T>
    inline payload_t<T> reduce(
        const detail::wrapper_t<T>& msg
      , const F& lambda = {}
      , const process_t root = process::root
      , const communicator_t& comm = world
    ) {
        auto out = (root == comm.rank)
            ? payload::create<T>(msg.count)
            : payload_t<T>();
        auto f = detail::collective::resolve_functor<T>(lambda);
        guard(MPI_Reduce(msg, out, msg.count, msg.type, f, root, comm));
        return out;
    }
}

namespace collective
{

    /**
     * Reduces generic messages into one process using an operator.
     * @tparam F The operator functor's implementation type.
     * @tparam T The message's contents type.
     * @param data The message to be reduced into the root process.
     * @param count The number of elements to be reduced.
     * @param lambda The operator functor to reduce messages with.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @return The resulting reduced message.
     */
    template <typename F, typename T>
    inline auto reduce(
        T *data
      , const size_t count
      , const F& lambda = {}
      , const process_t root = process::root
      , const communicator_t& comm = world
    ) {
        auto msg = detail::wrapper_t(data, count);
        return detail::collective::reduce<F>(msg, lambda, root, comm);
    }

    /**
     * Reduces containers into one process using an operator.
     * @tparam F The operator functor's implementation type.
     * @tparam T The type of container to be reduced.
     * @param data The container to be reduced into the root process.
     * @param lambda The operator functor to reduce messages with.
     * @param root The operation's root process.
     * @param comm The communicator this operation applies to.
     * @return The resulting reduced message.
     */
    template <typename F, typename T>
    inline auto reduce(
        T& data
      , const F& lambda = {}
      , const process_t root = process::root
      , const communicator_t& comm = world
    ) {
        auto msg = detail::wrapper_t(data);
        return detail::collective::reduce<F>(msg, lambda, root, comm);
    }
}

/*
 * Exposing the above-defined collective operation into the project's root namespace,
 * allowing it be called with decreased verbosity.
 */
using collective::reduce;

MPIWCPP17_END_NAMESPACE
