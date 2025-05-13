/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI send operation implementation detail.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2025-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/datatype.hpp>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/guard.hpp>
#include <mpiwcpp17/tag.hpp>

#include <mpiwcpp17/detail/payload.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail::operation
{
    /**
     * Sends a message to a process.
     * @tparam T The message payload type.
     * @param msg The message to send.
     * @param dest The message destination process.
     * @param tag The message identification tag.
     * @param comm The communicator this operation applies to.
     */
    template <typename T>
    MPIWCPP17_INLINE void send(
        const payload_in_t<T>& msg
      , const process_t dest
      , const tag_t tag
      , const communicator_t& comm
    ) {
        auto type = mpiwcpp17::datatype::identify<T>();
        auto mtag = tag >= 0 ? tag : mpiwcpp17::tag::ub;
        guard(MPI_Send(msg.ptr, msg.count, type, dest, mtag, comm));
    }
}

MPIWCPP17_END_NAMESPACE
