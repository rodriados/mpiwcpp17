/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI broadcast collective operation.
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
#include <mpiwcpp17/detail/operation/broadcast.hpp>

MPIWCPP17_BEGIN_NAMESPACE

inline namespace operation
{
    /**
     * Broadcasts a message in-place to all processes.
     * @tparam T The message payload type.
     * @param data The message data to broadcast.
     * @param count The number of elements to broadcast.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     */
    template <typename T>
    MPIWCPP17_INLINE void broadcast_inplace(
        T *data, size_t count
      , const process_t root = process::root
      , const communicator_t& comm = world
    ) {
        auto msg = detail::payload_in_t(data, count);
        detail::operation::broadcast_inplace(msg, root, comm);
    }

    /**
     * Broadcasts a message in-place to all processes.
     * @tparam T The message payload type.
     * @param data The message data to broadcast.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     */
    template <typename T>
    MPIWCPP17_INLINE void broadcast_inplace(
        T& data
      , const process_t root = process::root
      , const communicator_t& comm = world
    ) {
        auto msg = detail::payload::to_input(data);
        detail::operation::broadcast_inplace(msg, root, comm);
    }

    /**
     * Broadcasts a message to all processes.
     * @tparam T The message payload type.
     * @param data The message data to broadcast.
     * @param count The number of elements to broadcast.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     * @return The resulting broadcast message.
     */
    template <typename T>
    MPIWCPP17_INLINE auto broadcast(
        T *data, size_t count
      , const process_t root = process::root
      , const communicator_t& comm = world
    ) {
        broadcast_inplace(&count, 1, root, comm);
        auto msg = detail::payload_in_t(data, count);
        return detail::operation::broadcast(msg, root, comm);
    }

    /**
     * Broadcasts a message to all processes.
     * @tparam T The message payload type.
     * @param data The message data to broadcast.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     * @return The resulting broadcast message.
     */
    template <typename T>
    MPIWCPP17_INLINE auto broadcast(
        T& data
      , const process_t root = process::root
      , const communicator_t& comm = world
    ) {
        auto msg = detail::payload::to_input(data);
        if constexpr (detail::payload::is_contiguous_iterable_v<T>)
            return broadcast(msg.ptr, msg.count, root, comm);
        return detail::operation::broadcast(msg, root, comm);
    }
}

MPIWCPP17_END_NAMESPACE
