/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI send operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/global.hpp>
#include <mpiwcpp17/tag.hpp>

#include <mpiwcpp17/detail/payload.hpp>
#include <mpiwcpp17/detail/operation/send.hpp>

MPIWCPP17_BEGIN_NAMESPACE

inline namespace operation
{
    /**
     * Sends a message to a process.
     * @tparam T The message payload type.
     * @param data The message data to send.
     * @param count The number of elements to send.
     * @param dest The message destination process.
     * @param tag The message identification tag.
     * @param comm The communicator this operation applies to.
     */
    template <typename T>
    MPIWCPP17_INLINE void send(
        T *data, size_t count
      , const process_t dest = process::root
      , const tag_t tag = mpiwcpp17::tag::any
      , const communicator_t& comm = world
    ) {
        auto msg = detail::payload_in_t(data, count);
        detail::operation::send(msg, dest, tag, comm);
    }

    /**
     * Sends a message to a process.
     * @tparam T The message payload type.
     * @param data The message data to send.
     * @param dest The message destination process.
     * @param tag The message identification tag.
     * @param comm The communicator this operation applies to.
     */
    template <typename T>
    MPIWCPP17_INLINE void send(
        T& data
      , const process_t dest = process::root
      , const tag_t tag = mpiwcpp17::tag::any
      , const communicator_t& comm = world
    ) {
        auto msg = detail::payload::to_input(data);
        detail::operation::send(msg, dest, tag, comm);
    }
}

MPIWCPP17_END_NAMESPACE
