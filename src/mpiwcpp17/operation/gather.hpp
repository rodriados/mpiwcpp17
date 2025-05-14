/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI gather collective operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>
#include <numeric>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/policy.hpp>
#include <mpiwcpp17/global.hpp>

#include <mpiwcpp17/detail/payload.hpp>
#include <mpiwcpp17/detail/utility.hpp>
#include <mpiwcpp17/detail/operation/gather.hpp>

MPIWCPP17_BEGIN_NAMESPACE

inline namespace operation
{
    /**
     * Gathers a message to a process.
     * @tparam P The behavior policy guarantee.
     * @tparam T The message payload type.
     * @param data The message data to gather.
     * @param count The number of elements to gather.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     * @param policy The behavior policy guarantee.
     * @return The resulting gathered message.
     */
    template <typename P = policy::automatic_t, typename T>
    MPIWCPP17_INLINE auto gather(
        T *data, size_t count
      , const process_t root = process::root
      , const communicator_t& comm = world
      , const P policy = {}
    ) {
        auto msg = detail::payload_in_t(data, count);
        return detail::operation::gather(msg, root, comm, policy);
    }

    /**
     * Gathers a message to a process.
     * @tparam P The behavior policy guarantee.
     * @tparam T The message payload type.
     * @param data The message data to gather.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     * @param policy The behavior policy guarantee.
     * @return The resulting gathered message.
     */
    template <typename P = policy::automatic_t, typename T>
    MPIWCPP17_INLINE auto gather(
        T& data
      , const process_t root = process::root
      , const communicator_t& comm = world
      , const P policy = {}
    ) {
        auto msg = detail::payload::to_input(data);
        return detail::operation::gather(msg, root, comm, policy);
    }

    /**
     * Gathers a non-uniform message to a process.
     * @tparam T The message payload type.
     * @param data The message data to gather.
     * @param count The number of elements to gather from current process.
     * @param counts The number of elements to gather by process.
     * @param displs The displacement of each process message.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    MPIWCPP17_INLINE auto gather(
        T *data, size_t count
      , const detail::capture_t<int>& counts
      , const detail::capture_t<int>& displs
      , const process_t root = process::root
      , const communicator_t& comm = world
    ) {
        auto msg = detail::payload_in_t(data, count);
        auto total = std::accumulate(counts.begin(), counts.end(), 0);
        return detail::operation::gatherv(msg, counts, displs, total, root, comm);
    }

    /**
     * Gathers a non-uniform message to a process.
     * @tparam T The message payload type.
     * @param data The message data to gather.
     * @param counts The number of elements to gather by process.
     * @param displs The displacement of each process message.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    MPIWCPP17_INLINE auto gather(
        T& data
      , const detail::capture_t<int>& counts
      , const detail::capture_t<int>& displs
      , const process_t root = process::root
      , const communicator_t& comm = world
    ) {
        auto msg = detail::payload::to_input(data);
        auto total = std::accumulate(counts.begin(), counts.end(), 0);
        return detail::operation::gatherv(msg, counts, displs, total, root, comm);
    }

    /**
     * Gathers a uniform message in-place to a process.
     * @tparam T The message payload type.
     * @param data The message data to gather.
     * @param count The number of elements to gather.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     */
    template <typename T>
    MPIWCPP17_INLINE void gather_inplace(
        T *data, size_t count
      , const process_t root = process::root
      , const communicator_t& comm = world
    ) {
        auto msg = detail::payload_in_t(data, count);
        detail::operation::gather_inplace(msg, root, comm);
    }

    /**
     * Gathers a non-uniform message in-place to a process.
     * @tparam T The message payload type.
     * @param data The message data to gather.
     * @param counts The number of elements to gather by process.
     * @param displs The displacement of each process message.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     */
    template <typename T>
    MPIWCPP17_INLINE void gather_inplace(
        T *data
      , const detail::capture_t<int>& counts
      , const detail::capture_t<int>& displs
      , const process_t root = process::root
      , const communicator_t& comm = world
    ) {
        auto msg = detail::payload_in_t(data, 0);
        detail::operation::gatherv_inplace(msg, counts, displs, root, comm);
    }
}

MPIWCPP17_END_NAMESPACE
