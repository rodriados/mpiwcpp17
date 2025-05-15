/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI scatter operation implementation detail.
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
#include <mpiwcpp17/detail/operation/broadcast.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail::operation
{
    /**
     * Scatters a non-uniform message in-place from a process.
     * @tparam T The message payload type.
     * @param msg The message to scatter in-place.
     * @param counts The number of elements to scatter to each process.
     * @param displs The displacement of each process message.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     */
    template <typename T>
    MPIWCPP17_INLINE void scatterv_inplace(
        const payload_in_t<T>& msg
      , const payload_const_t<int>& counts
      , const payload_const_t<int>& displs
      , const process_t root
      , const communicator_t& comm
    ) {
        auto type = mpiwcpp17::datatype::identify<T>();
        MPIWCPP17_GUARD_EVAL(
            (root == communicator::rank(comm))
              ? MPI_Scatterv(msg.ptr, counts, displs, type, MPI_IN_PLACE, 0, type, root, comm)
              : MPI_Scatterv(nullptr, counts, displs, type, msg.ptr, msg.count, type, root, comm));
    }

    /**
     * Scatters a uniform message in-place from a process.
     * @tparam T The message payload type.
     * @param msg The message to scatter in-place.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     */
    template <typename T>
    MPIWCPP17_INLINE void scatter_inplace(
        const payload_in_t<T>& msg
      , const process_t root
      , const communicator_t& comm
    ) {
        auto type = mpiwcpp17::datatype::identify<T>();
        MPIWCPP17_GUARD_EVAL(
            (root == communicator::rank(comm))
              ? MPI_Scatter(msg.ptr, msg.count, type, MPI_IN_PLACE, 0, type, root, comm)
              : MPI_Scatter(nullptr, 0, type, msg.ptr, msg.count, type, root, comm));
    }

    /**
     * Scatters a non-uniform message from a process.
     * @tparam T The message payload type.
     * @param msg The message to scatter.
     * @param counts The number of elements to scatter to each process.
     * @param displs The displacement of each process message.
     * @param total The total number of elements to allocate.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     * @return The resulting scattered message.
     */
    template <typename T>
    MPIWCPP17_INLINE auto scatterv(
        const payload_in_t<T>& msg
      , const payload_const_t<int>& counts
      , const payload_const_t<int>& displs
      , const size_t total
      , const process_t root
      , const communicator_t& comm
    ) {
        auto type = mpiwcpp17::datatype::identify<T>();
        auto out = payload::create_output<T>(total);
        guard(MPI_Scatterv(msg.ptr, counts, displs, type, out, total, type, root, comm));
        return out;
    }

    /**
     * Scatters a uniform message from a process.
     * @tparam T The message payload type.
     * @param msg The message to scatter.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     * @return The resulting scattered message.
     */
    template <typename T>
    MPIWCPP17_INLINE auto scatter(
        const payload_in_t<T>& msg
      , const process_t root
      , const communicator_t& comm
      , const policy::uniform_t
    ) {
        auto type = mpiwcpp17::datatype::identify<T>();
        auto out = payload::create_output<T>(msg.count / communicator::size(comm));
        guard(MPI_Scatter(msg.ptr, out.count, type, out, out.count, type, root, comm));
        return out;
    }

    /**
     * Scatters a non-uniform message from a process.
     * @tparam T The message payload type.
     * @param msg The message to scatter.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     * @return The resulting scattered message.
     */
    template <typename T>
    MPIWCPP17_INLINE auto scatter(
        const payload_in_t<T>& msg
      , const process_t root
      , const communicator_t& comm
      , const policy::varying_t
    ) {
        size_t rank  = communicator::rank(comm);
        size_t size  = communicator::size(comm);

        size_t quotient  = msg.count / size;
        size_t remainder = msg.count % size;

        if (remainder == 0)
            return scatter(msg, root, comm, policy::uniform);

        auto total   = quotient + (remainder > rank);
        bool is_root = rank == static_cast<size_t>(root);

        auto counts  = payload::create_output<int>(is_root * size);
        auto displs  = payload::create_output<int>(is_root * size);

        if (is_root) for (size_t j = 0; j < size; ++j) {
            counts[j] = quotient + (remainder > j);
            displs[j] = quotient * j + std::min(j, remainder);
        }

        return scatterv(msg, counts, displs, total, root, comm);
    }

    /**
     * Scatters a message without known policy from a process.
     * @tparam T The message payload type.
     * @param msg The message to scatter.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     * @return The resulting scattered message.
     */
    template <typename T>
    MPIWCPP17_INLINE auto scatter(
        const payload_in_t<T>& msg
      , const process_t root
      , const communicator_t& comm
      , const policy::automatic_t
    ) {
        // As the non-uniform case already needs to detect and falls back to uniform
        // implementation when appropriate, there is nothing extra we can do to improve
        // performance in a `scatter` operation with automatic policy.
        return scatter(msg, root, comm, policy::varying);
    }
}

MPIWCPP17_END_NAMESPACE
