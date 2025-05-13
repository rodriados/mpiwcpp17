/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI broadcast operation implementation detail.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2025-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/datatype.hpp>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/policy.hpp>
#include <mpiwcpp17/guard.hpp>

#include <mpiwcpp17/detail/payload.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail::operation
{
    /**
     * Broadcasts a message in-place to all processes.
     * @tparam T The message payload type.
     * @param msg The message to broadcast in-place.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     */
    template <typename T>
    MPIWCPP17_INLINE void broadcast_inplace(
        const payload_in_t<T>& msg
      , const process_t root
      , const communicator_t& comm
    ) {
        const auto type = mpiwcpp17::datatype::identify<T>();
        guard(MPI_Bcast(msg.ptr, msg.count, type, root, comm));
    }

    /**
     * Broadcasts a uniform message to all processes.
     * @tparam T The message payload type.
     * @param msg The message to broadcast.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     * @return The resulting broadcast message.
     */
    template <typename T>
    MPIWCPP17_INLINE auto broadcast(
        const payload_in_t<T>& msg
      , const process_t root
      , const communicator_t& comm
      , const policy::uniform_t
    ) {
        auto out = (root == communicator::rank(comm))
          ? payload::copy_to_output(msg)
          : payload::create_output<T>(msg.count);
        broadcast_inplace(out, root, comm);
        return out;
    }

    /**
     * Broadcasts a message without known policy to all processes.
     * @tparam T The message payload type.
     * @param msg The message to broadcast.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     * @return The resulting broadcast message.
     */
    template <typename T>
    MPIWCPP17_INLINE auto broadcast(
        const payload_in_t<T>& msg
      , const process_t root
      , const communicator_t& comm
      , const policy::automatic_t
    ) {
        size_t count = msg.count;
        broadcast_inplace<size_t>(&count, root, comm);
        return broadcast<T>({msg.ptr, count}, root, comm, policy::uniform);
    }
}

MPIWCPP17_END_NAMESPACE
