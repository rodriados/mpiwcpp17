/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI receive operation implementation detail.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2025-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>
#include <utility>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/datatype.hpp>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/status.hpp>
#include <mpiwcpp17/guard.hpp>
#include <mpiwcpp17/tag.hpp>

#include <mpiwcpp17/detail/payload.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail::operation
{
    namespace datatype = mpiwcpp17::datatype;

    /**
     * Receives a message in-place from a process.
     * @tparam T The message payload type.
     * @param msg The message to receive in-place.
     * @param source The message source process.
     * @param tag The message identification tag.
     * @param comm The communicator this operation applies to.
     * @return The resulting operation status.
     */
    template <typename T>
    MPIWCPP17_INLINE auto receive_inplace(
        const payload_in_t<T>& msg
      , const process_t source
      , const tag_t tag
      , const communicator_t& comm
    ) {
        auto type = datatype::identify<T>();
        return MPIWCPP17_GUARD_CALL(
            status_t
          , MPI_Recv(msg.ptr, msg.count, type, source, tag, comm, &_)
        );
    }

    /**
     * Receives a message from a process.
     * @tparam T The message payload type.
     * @param count The number of elements to receive.
     * @param source The message source process.
     * @param tag The message identification tag.
     * @param comm The communicator this operation applies to.
     * @return The resulting operation status and received message.
     */
    template <typename T>
    MPIWCPP17_INLINE auto receive(
        const size_t count
      , const process_t source
      , const tag_t tag
      , const communicator_t& comm
    ) {
        auto out = payload::create_output<T>(count);
        auto status = receive_inplace(out, source, tag, comm);
        return std::make_pair(status, std::move(out));
    }
}

MPIWCPP17_END_NAMESPACE
