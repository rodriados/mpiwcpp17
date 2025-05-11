/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI all-reduce operation implementation detail.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2025-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/datatype.hpp>
#include <mpiwcpp17/guard.hpp>

#include <mpiwcpp17/detail/functor.hpp>
#include <mpiwcpp17/detail/payload.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail::operation
{
    namespace datatype = mpiwcpp17::datatype;

    /**
     * Reduces a message in-place to all processes.
     * @tparam T The message payload type.
     * @tparam F The reduce operator type.
     * @param msg The message to reduce in-place.
     * @param lambda The functor to reduce message with.
     * @param comm The communicator the operation applies to.
     */
    template <typename T, typename F>
    MPIWCPP17_INLINE void allreduce_inplace(
        const payload_in_t<T>& msg
      , const F& lambda
      , const communicator_t& comm
    ) {
        auto type = datatype::identify<T>();
        auto f = functor::resolve<T>(lambda);
        guard(MPI_Allreduce(MPI_IN_PLACE, msg.ptr, msg.count, type, f, comm));
    }

    /**
     * Reduces a message to all processes.
     * @tparam T The message payload type.
     * @tparam F The reduce operator type.
     * @param msg The message to reduce.
     * @param lambda The functor to reduce message with.
     * @param root The operation root process.
     * @param comm The communicator the operation applies to.
     * @return The resulting reduced message.
     */
    template <typename T, typename F>
    MPIWCPP17_INLINE auto allreduce(
        const payload_in_t<T>& msg
      , const F& lambda
      , const communicator_t& comm
    ) {
        auto type = datatype::identify<T>();
        auto f = functor::resolve<T>(lambda);
        auto out = payload::create_output<T>(msg.count);
        guard(MPI_Allreduce(msg.ptr, out.ptr, msg.count, type, f, comm));
        return out;
    }
}

MPIWCPP17_END_NAMESPACE
