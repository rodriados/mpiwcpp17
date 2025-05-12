/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI all-gather collective operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>
#include <numeric>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/policy.hpp>
#include <mpiwcpp17/global.hpp>

#include <mpiwcpp17/detail/payload.hpp>
#include <mpiwcpp17/detail/utility.hpp>
#include <mpiwcpp17/detail/operation/allgather.hpp>

MPIWCPP17_BEGIN_NAMESPACE

inline namespace operation
{
    /**
     * Gathers a non-uniform message in-place to all processes.
     * @tparam T The message payload type.
     * @param data The message data to gather.
     * @param count The number of elements to gather by process.
     * @param displ The displacement of each process message.
     * @param comm The communicator the operation applies to.
     */
    template <typename T>
    MPIWCPP17_INLINE void allgather_inplace(
        T *data
      , const detail::iterable_t<int>& count
      , const detail::iterable_t<int>& displ
      , const communicator_t& comm = world
    ) {
        auto msg = detail::payload_in_t(data, 0);
        detail::operation::allgatherv_inplace(msg, count, displ, comm);
    }

    /**
     * Gathers a uniform message in-place to all processes.
     * @tparam T The message payload type.
     * @param data The message data to gather.
     * @param count The number of elements to gather.
     * @param comm The communicator the operation applies to.
     */
    template <typename T>
    MPIWCPP17_INLINE void allgather_inplace(
        T *data, size_t count
      , const communicator_t& comm = world
    ) {
        auto msg = detail::payload_in_t(data, count);
        detail::operation::allgather_inplace(msg, comm);
    }

    /**
     * Gathers a non-uniform message to all processes.
     * @tparam T The message payload type.
     * @param data The message data to gather.
     * @param count The number of elements to gather by process.
     * @param displ The displacement of each process message.
     * @param comm The communicator the operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    MPIWCPP17_INLINE auto allgather(
        T *data
      , const detail::iterable_t<int>& count
      , const detail::iterable_t<int>& displ
      , const communicator_t& comm = world
    ) {
        auto msg = detail::payload_in_t(data, 0);
        auto total = std::accumulate(count.begin(), count.end(), 0);
        return detail::operation::allgatherv(msg, count, displ, total, comm);
    }

    /**
     * Gathers a non-uniform message to all processes.
     * @tparam T The message payload type.
     * @param data The message data to gather.
     * @param count The number of elements to gather by process.
     * @param displ The displacement of each process message.
     * @param comm The communicator the operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    MPIWCPP17_INLINE auto allgather(
        T& data
      , const detail::iterable_t<int>& count
      , const detail::iterable_t<int>& displ
      , const communicator_t& comm = world
    ) {
        auto msg = detail::payload::to_input(data);
        auto total = std::accumulate(count.begin(), count.end(), 0);
        return detail::operation::allgatherv(msg, count, displ, total, comm);
    }

    /**
     * Gathers a message to all processes.
     * @tparam P The behavior policy guarantee.
     * @tparam T The message payload type.
     * @param data The message data to gather.
     * @param count The number of elements to gather.
     * @param comm The communicator the operation applies to.
     * @param policy The behavior policy guarantee.
     * @return The resulting gathered message.
     */
    template <typename P = policy::automatic_t, typename T>
    MPIWCPP17_INLINE auto allgather(
        T *data, size_t count
      , const communicator_t& comm = world
      , const P& policy = {}
    ) {
        auto msg = detail::payload_in_t(data, count);
        return detail::operation::allgather(msg, comm, policy);
    }

    /**
     * Gathers a message to all processes.
     * @tparam P The behavior policy guarantee.
     * @tparam T The message payload type.
     * @param data The message data to gather.
     * @param comm The communicator the operation applies to.
     * @param policy The behavior policy guarantee.
     * @return The resulting gathered message.
     */
    template <typename P = policy::automatic_t, typename T>
    MPIWCPP17_INLINE auto allgather(
        T& data
      , const communicator_t& comm = world
      , const P& policy = {}
    ) {
        auto msg = detail::payload::to_input(data);
        return detail::operation::allgather(msg, comm, policy);
    }
}

MPIWCPP17_END_NAMESPACE
