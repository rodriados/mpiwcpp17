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
#include <mpiwcpp17/global.hpp>

#include <mpiwcpp17/detail/payload.hpp>
#include <mpiwcpp17/detail/operation/allreduce.hpp>

MPIWCPP17_BEGIN_NAMESPACE

inline namespace operation
{
    /**
     * Reduces a message to all processes.
     * @tparam T The message payload type.
     * @tparam F The reduce operator type.
     * @param data The message data to reduce.
     * @param count The number of elements to reduce.
     * @param lambda The functor to reduce message with.
     * @param comm The communicator the operation applies to.
     * @return The resulting reduced message.
     */
    template <typename T, typename F>
    MPIWCPP17_INLINE auto allreduce(
        T *data, size_t count
      , const F& lambda
      , const communicator_t& comm = world
    ) {
        auto msg = detail::payload_in_t(data, count);
        return detail::operation::allreduce(msg, lambda, comm);
    }

    /**
     * Reduces a message to all processes.
     * @tparam T The message payload type.
     * @tparam F The reduce operator type.
     * @param data The message data to reduce.
     * @param lambda The functor to reduce message with.
     * @param comm The communicator the operation applies to.
     * @return The resulting reduced message.
     */
    template <typename T, typename F>
    MPIWCPP17_INLINE auto allreduce(
        T& data
      , const F& lambda
      , const communicator_t& comm = world
    ) {
        auto msg = detail::payload::to_input(data);
        return detail::operation::allreduce(msg, lambda, comm);
    }

    /**
     * Reduces a message in-place to all processes.
     * @tparam T The message payload type.
     * @tparam F The reduce operator type.
     * @param data The message data to reduce.
     * @param count The number of elements to reduce.
     * @param lambda The functor to reduce message with.
     * @param comm The communicator the operation applies to.
     */
    template <typename T, typename F>
    MPIWCPP17_INLINE void allreduce_inplace(
        T *data, size_t count
      , const F& lambda
      , const communicator_t& comm = world
    ) {
        auto msg = detail::payload_in_t(data, count);
        detail::operation::allreduce_inplace(msg, lambda, comm);
    }

    /**
     * Reduces a message in-place to all processes.
     * @tparam T The message payload type.
     * @tparam F The reduce operator type.
     * @param data The message data to reduce.
     * @param lambda The functor to reduce message with.
     * @param comm The communicator the operation applies to.
     */
    template <typename T, typename F>
    MPIWCPP17_INLINE void allreduce_inplace(
        T& data
      , const F& lambda
      , const communicator_t& comm = world
    ) {
        auto msg = detail::payload::to_input(data);
        detail::operation::allreduce_inplace(msg, lambda, comm);
    }
}

MPIWCPP17_END_NAMESPACE
