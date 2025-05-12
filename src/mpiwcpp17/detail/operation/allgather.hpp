/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI all-gather operation implementation detail.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2025-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/datatype.hpp>
#include <mpiwcpp17/functor.hpp>
#include <mpiwcpp17/policy.hpp>
#include <mpiwcpp17/guard.hpp>

#include <mpiwcpp17/detail/payload.hpp>
#include <mpiwcpp17/detail/operation/allreduce.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail::operation
{
    namespace datatype = mpiwcpp17::datatype;

    /**
     * Gathers a non-uniform message in-place to all processes.
     * @tparam T The message payload type.
     * @param msg The message to gather in-place.
     * @param count The number of elements to gather by process.
     * @param displ The displacement of each process message.
     * @param comm The communicator the operation applies to.
     */
    template <typename T>
    MPIWCPP17_INLINE void allgatherv_inplace(
        const payload_in_t<T>& msg
      , const payload_const_t<int>& count
      , const payload_const_t<int>& displ
      , const communicator_t& comm
    ) {
        auto type = datatype::identify<T>();
        guard(MPI_Allgatherv(MPI_IN_PLACE, 0, type, msg.ptr, count, displ, type, comm));
    }

    /**
     * Gathers a uniform message in-place to all processes.
     * @tparam T The message payload type.
     * @param msg The message to gather in-place.
     * @param comm The communicator the operation applies to.
     */
    template <typename T>
    MPIWCPP17_INLINE void allgather_inplace(
        const payload_in_t<T>& msg
      , const communicator_t& comm
    ) {
        auto type = datatype::identify<T>();
        guard(MPI_Allgather(MPI_IN_PLACE, 0, type, msg.ptr, msg.count, type, comm));
    }

    /**
     * Gathers a non-uniform message to all processes.
     * @tparam T The message payload type.
     * @param msg The message to gather.
     * @param count The number of elements to gather by process.
     * @param displ The displacement of each process message.
     * @param total The total number of elements to allocate.
     * @param comm The communicator the operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    MPIWCPP17_INLINE auto allgatherv(
        const payload_in_t<T>& msg
      , const payload_const_t<int>& count
      , const payload_const_t<int>& displ
      , const size_t total
      , const communicator_t& comm
    ) {
        auto type = datatype::identify<T>();
        auto out = payload::create_output<T>(total);
        auto msglen = msg.count ? msg.count : count[communicator::rank(comm)];
        guard(MPI_Allgatherv(msg.ptr, msglen, type, out, count, displ, type, comm));
        return out;
    }

    /**
     * Gathers a uniform message to all processes.
     * @tparam T The message payload type.
     * @param msg The message to gather.
     * @param comm The communicator the operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    MPIWCPP17_INLINE auto allgather(
        const payload_in_t<T>& msg
      , const communicator_t& comm
      , const policy::uniform_t&
    ) {
        auto type = datatype::identify<T>();
        auto out = payload::create_output<T>(msg.count * communicator::size(comm));
        guard(MPI_Allgather(msg.ptr, msg.count, type, out.ptr, msg.count, type, comm));
        return out;
    }

    /**
     * Gathers a non-uniform message to all processes.
     * @tparam T The message payload type.
     * @param msg The message to gather.
     * @param comm The communicator the operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    MPIWCPP17_INLINE auto allgather(
        const payload_in_t<T>& msg
      , const communicator_t& comm
      , const policy::varying_t&
    ) {
        int total  = 0;
        int mcount = static_cast<int>(msg.count);
        int size   = communicator::size(comm);

        auto count = allgather<int>({&mcount, 1}, comm, policy::uniform);
        auto displ = payload::create_output<int>(size);
        for (int j = 0; j < size; total += count[j++])
            displ[j] = total;
        return allgatherv(msg, count, displ, total, comm);
    }

    /**
     * Gathers a message without known policy to all processes.
     * @tparam T The message payload type.
     * @param msg The message to gather.
     * @param comm The communicator the operation applies to.
     * @return The resulting gathered message.
     */
    template <typename T>
    MPIWCPP17_INLINE auto allgather(
        const payload_in_t<T>& msg
      , const communicator_t& comm
      , const policy::automatic_t&
    ) {
        int64_t count[2] = {
            +int64_t(msg.count)
          , -int64_t(msg.count)};
        auto& min = mpiwcpp17::functor::min;
        allreduce_inplace<int64_t>({count, 2}, min, comm);
        return (+count[0] == -count[1])
          ? allgather(msg, comm, policy::uniform)
          : allgather(msg, comm, policy::varying);
    }
}

MPIWCPP17_END_NAMESPACE
