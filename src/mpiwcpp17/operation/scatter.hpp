/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI scatter collective operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/policy.hpp>
#include <mpiwcpp17/global.hpp>

#include <mpiwcpp17/detail/payload.hpp>
#include <mpiwcpp17/detail/utility.hpp>
#include <mpiwcpp17/detail/operation/scatter.hpp>
#include <mpiwcpp17/operation/broadcast.hpp>

MPIWCPP17_BEGIN_NAMESPACE

inline namespace operation
{
    /**
     * Scatters a uniform message from a process.
     * @tparam P The behavior policy guarantee.
     * @tparam T The message payload type.
     * @param data The message data to scatter.
     * @param count The number of elements to scatter.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     * @param policy The behavior policy guarantee.
     * @return The resulting scattered message.
     */
    template <typename P = policy::automatic_t, typename T>
    MPIWCPP17_INLINE auto scatter(
        T *data, size_t count
      , const process_t root = process::root
      , const communicator_t& comm = world
      , const P policy = {}
    ) {
        broadcast_inplace(count, root, comm);
        auto msg = detail::payload_in_t(data, count);
        return detail::operation::scatter(msg, root, comm, policy);
    }

    /**
     * Scatters a uniform message from a process.
     * @tparam P The behavior policy guarantee.
     * @tparam T The message payload type.
     * @param data The message data to scatter.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     * @param policy The behavior policy guarantee.
     * @return The resulting scattered message.
     */
    template <typename P = policy::automatic_t, typename T>
    MPIWCPP17_INLINE auto scatter(
        T& data
      , const process_t root = process::root
      , const communicator_t& comm = world
      , const P policy = {}
    ) {
        auto [ptr, count] = detail::payload::to_tentative_input(data);
        return scatter(ptr, count, root, comm, policy);
    }

    /**
     * Scatters a non-uniform message from a process.
     * @tparam T The message payload type.
     * @param data The message data to scatter.
     * @param count The number of elements to scatter.
     * @param counts The number of elements to scatter to each process.
     * @param displs The displacement of each process message.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     * @return The resulting scattered message.
     */
    template <typename T>
    MPIWCPP17_INLINE auto scatter(
        T *data, size_t count
      , const detail::capture_t<int>& counts
      , const detail::capture_t<int>& displs
      , const process_t root = process::root
      , const communicator_t& comm = world
    ) {
        auto msg = detail::payload_in_t(data, count);
        auto total = counts[communicator::rank(comm)];
        return detail::operation::scatterv(msg, counts, displs, total, root, comm);
    }

    /**
     * Scatters a non-uniform message from a process.
     * @tparam T The message payload type.
     * @param data The message data to scatter.
     * @param counts The number of elements to scatter to each process.
     * @param displs The displacement of each process message.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     * @return The resulting scattered message.
     */
    template <typename T>
    MPIWCPP17_INLINE auto scatter(
        T& data
      , const detail::capture_t<int>& counts
      , const detail::capture_t<int>& displs
      , const process_t root = process::root
      , const communicator_t& comm = world
    ) {
        auto msg = detail::payload::to_input(data);
        auto total = counts[communicator::rank(comm)];
        return detail::operation::scatterv(msg, counts, displs, total, root, comm);
    }

    /**
     * Scatters a uniform message in-place from a process.
     * @tparam T The message payload type.
     * @param data The message data to scatter.
     * @param count The number of elements to scatter.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     */
    template <typename T>
    MPIWCPP17_INLINE void scatter_inplace(
        T *data, size_t count
      , const process_t root = process::root
      , const communicator_t& comm = world
    ) {
        auto msg = detail::payload_in_t(data, count);
        detail::operation::scatter_inplace(msg, root, comm);
    }

    /**
     * Scatters a non-uniform message in-place from a process.
     * @tparam T The message payload type.
     * @param data The message data to scatter.
     * @param count The number of elements to scatter.
     * @param counts The number of elements to scatter to each process.
     * @param displs The displacement of each process message.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     */
    template <typename T>
    MPIWCPP17_INLINE void scatter_inplace(
        T *data, size_t count
      , const detail::capture_t<int>& counts
      , const detail::capture_t<int>& displs
      , const process_t root = process::root
      , const communicator_t& comm = world
    ) {
        auto msg = detail::payload_in_t(data, count);
        detail::operation::scatterv_inplace(msg, counts, displs, root, comm);
    }
}

MPIWCPP17_END_NAMESPACE
