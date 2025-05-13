/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI receive operation.
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
#include <mpiwcpp17/detail/operation/receive.hpp>
#include <mpiwcpp17/operation/probe.hpp>

MPIWCPP17_BEGIN_NAMESPACE

inline namespace operation
{
    /**
     * Receives a message from a process.
     * @tparam T The message payload type.
     * @param source The message source process.
     * @param tag The message identification tag.
     * @param comm The communicator this operation applies to.
     * @return The resulting operation status and received message.
     */
    template <typename T>
    MPIWCPP17_INLINE auto receive(
        const process_t source = process::any
      , const tag_t tag = mpiwcpp17::tag::any
      , const communicator_t& comm = world
    ) {
        auto status = operation::probe(source, tag, comm);
        auto count  = mpiwcpp17::status::count<T>(status);
        return detail::operation::receive<T>(count, source, tag, comm);
    }

    /**
     * Receives a message in-place from a process.
     * @tparam T The message payload type.
     * @param data The message data to receive.
     * @param count The number of elements to receive.
     * @param source The message source process.
     * @param tag The message identification tag.
     * @param comm The communicator this operation applies to.
     * @return The resulting operation status.
     */
    template <typename T>
    MPIWCPP17_INLINE auto receive_inplace(
        T *data, size_t count
      , const process_t source = process::any
      , const tag_t tag = mpiwcpp17::tag::any
      , const communicator_t& comm = world
    ) {
        auto msg = detail::payload_in_t(data, count);
        return detail::operation::receive_inplace(msg, source, tag, comm);
    }

    /**
     * Receives a message in-place from a process.
     * @tparam T The message payload type.
     * @param data The message data to receive.
     * @param source The message source process.
     * @param tag The message identification tag.
     * @param comm The communicator this operation applies to.
     * @return The resulting operation status.
     */
    template <typename T>
    MPIWCPP17_INLINE auto receive_inplace(
        T& data
      , const process_t source = process::any
      , const tag_t tag = mpiwcpp17::tag::any
      , const communicator_t& comm = world
    ) {
        auto msg = detail::payload::to_input(data);
        return detail::operation::receive_inplace(msg, source, tag, comm);
    }
}

MPIWCPP17_END_NAMESPACE
