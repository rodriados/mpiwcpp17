/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI reduce collective operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/global.hpp>

#include <mpiwcpp17/detail/payload.hpp>
#include <mpiwcpp17/detail/operation/reduce.hpp>

MPIWCPP17_BEGIN_NAMESPACE

inline namespace operation
{
    /**
     * Reduces a message to a process.
     * @tparam T The message payload type.
     * @tparam F The reduce operator type.
     * @param data The message data to reduce.
     * @param count The number of elements to reduce.
     * @param lambda The functor to reduce message with.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     * @return The resulting reduced message.
     */
    template <typename T, typename F>
    MPIWCPP17_INLINE auto reduce(
        T* data, size_t count
      , const F& lambda
      , const process_t root = process::root
      , const communicator_t& comm = world
    ) {
        auto msg = detail::payload_in_t(data, count);
        return detail::operation::reduce(msg, lambda, root, comm);
    }

    /**
     * Reduces a message to a process.
     * @tparam T The message payload type.
     * @tparam F The reduce operator type.
     * @param data The message data to reduce.
     * @param lambda The functor to reduce message with.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     * @return The resulting reduced message.
     */
    template <typename T, typename F>
    MPIWCPP17_INLINE auto reduce(
        T& data
      , const F& lambda
      , const process_t root = process::root
      , const communicator_t& comm = world
    ) {
        auto msg = detail::payload::to_input(data);
        return detail::operation::reduce(msg, lambda, root, comm);
    }

    /**
     * Reduces a message in-place to a process.
     * @tparam T The message payload type.
     * @tparam F The reduce operator type.
     * @param data The message data to reduce.
     * @param count The number of elements to reduce.
     * @param lambda The functor to reduce message with.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     */
    template <typename T, typename F>
    MPIWCPP17_INLINE void reduce_inplace(
        T* data, size_t count
      , const F& lambda
      , const process_t root = process::root
      , const communicator_t& comm = world
    ) {
        auto msg = detail::payload_in_t(data, count);
        detail::operation::reduce_inplace(msg, lambda, root, comm);
    }

    /**
     * Reduces a message in-place to a process.
     * @tparam T The message payload type.
     * @tparam F The reduce operator type.
     * @param data The message data to reduce.
     * @param lambda The functor to reduce message with.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     */
    template <typename T, typename F>
    MPIWCPP17_INLINE void reduce_inplace(
        T& data
      , const F& lambda
      , const process_t root = process::root
      , const communicator_t& comm = world
    ) {
        auto msg = detail::payload::to_input(data);
        detail::operation::reduce_inplace(msg, lambda, root, comm);
    }
}

MPIWCPP17_END_NAMESPACE
