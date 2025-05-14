/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI gather operation implementation detail.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2025-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>
#include <numeric>

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
     * Gathers a non-uniform message in-place to a process.
     * @tparam T The message payload type.
     * @param msg The message to gather in-place.
     * @param counts The number of elements to gather by process.
     * @param displs The displacement of each process message.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     */
    template <typename T>
    MPIWCPP17_INLINE void gatherv_inplace(
        const payload_in_t<T>& msg
      , const payload_const_t<int>& counts
      , const payload_const_t<int>& displs
      , const process_t root
      , const communicator_t& comm
    ) {
        auto type = mpiwcpp17::datatype::identify<T>();
        auto count = counts[communicator::rank(comm)];
        MPIWCPP17_GUARD_EVAL(
            (root == communicator::rank(comm))
              ? MPI_Gatherv(MPI_IN_PLACE, 0, type, msg.ptr, counts, displs, type, root, comm)
              : MPI_Gatherv(msg.ptr, count, type, nullptr, counts, displs, type, root, comm));
    }

    /**
     * Gathers a uniform message in-place to a process.
     * @tparam T The message payload type.
     * @param msg The message to gather in-place.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     */
    template <typename T>
    MPIWCPP17_INLINE void gather_inplace(
        const payload_in_t<T>& msg
      , const process_t root
      , const communicator_t& comm
    ) {
        auto type = mpiwcpp17::datatype::identify<T>();
        MPIWCPP17_GUARD_EVAL(
            (root == communicator::rank(comm))
              ? MPI_Gather(MPI_IN_PLACE, 0, type, msg.ptr, msg.count, type, root, comm)
              : MPI_Gather(msg.ptr, msg.count, type, nullptr, 0, type, root, comm));
    }

    /**
     * Gathers a non-uniform message to a process.
     * @tparam T The message payload type.
     * @param msg The message to gather.
     * @param counts The number of elements to gather by process.
     * @param displs The displacement of each process message.
     * @param total The total number of elements to allocate.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    MPIWCPP17_INLINE auto gatherv(
        const payload_in_t<T>& msg
      , const payload_const_t<int>& counts
      , const payload_const_t<int>& displs
      , const size_t total
      , const process_t root
      , const communicator_t& comm
    ) {
        auto type = mpiwcpp17::datatype::identify<T>();
        auto out = (root == communicator::rank(comm))
          ? payload::create_output<T>(total)
          : payload::create_output<T>(0);
        guard(MPI_Gatherv(msg.ptr, msg.count, type, out, counts, displs, type, root, comm));
        return out;
    }

    /**
     * Gathers a uniform message to a process.
     * @tparam T The message payload type.
     * @param msg The message to gather.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    MPIWCPP17_INLINE auto gather(
        const payload_in_t<T>& msg
      , const process_t root
      , const communicator_t& comm
      , const policy::uniform_t
    ) {
        auto type = mpiwcpp17::datatype::identify<T>();
        auto out = (root == communicator::rank(comm))
          ? payload::create_output<T>(msg.count * communicator::size(comm))
          : payload::create_output<T>(0);
        guard(MPI_Gather(msg.ptr, msg.count, type, out, msg.count, type, root, comm));
        return out;
    }

    /**
     * Gathers a non-uniform message to a process.
     * @tparam T The message payload type.
     * @param msg The message to gather.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    MPIWCPP17_INLINE auto gather(
        const payload_in_t<T>& msg
      , const process_t root
      , const communicator_t& comm
      , const policy::varying_t
    ) {
        auto count  = static_cast<int>(msg.count);
        auto counts = gather<int>(&count, root, comm, policy::uniform);
        auto displs = payload::create_output<int>(counts.count);

        std::exclusive_scan(counts.begin(), counts.end(), displs.begin(), 0);
        size_t total = !counts ? 0 : displs.last() + counts.last();

        return gatherv(msg, counts, displs, total, root, comm);
    }

    /**
     * Gathers a message without known policy to a process.
     * @tparam T The message payload type.
     * @param msg The message to gather.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    MPIWCPP17_INLINE auto gather(
        const payload_in_t<T>& msg
      , const process_t root
      , const communicator_t& comm
      , const policy::automatic_t
    ) {
        // As there is no efficient way for non-root processes to know whether the
        // payload is uniform or not, we assume that it is always non-uniform unless
        // explicitly informed by the user. We prefer paying a small overhead for
        // always calling `gatherv`, than another `broadcast` for correctness.
        return gather(msg, root, comm, policy::varying);
    }
}

MPIWCPP17_END_NAMESPACE
