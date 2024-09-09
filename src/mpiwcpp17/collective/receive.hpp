/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI receive collective operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>
#include <utility>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/status.hpp>
#include <mpiwcpp17/global.hpp>
#include <mpiwcpp17/guard.hpp>
#include <mpiwcpp17/tag.hpp>

#include <mpiwcpp17/collective/probe.hpp>
#include <mpiwcpp17/detail/payload.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail::collective
{
    /**
     * Waits and receives a message from a process connected to a communicator.
     * @tparam T The message's contents type.
     * @param count The number of elements in message to be received.
     * @param source The process to receive the message from.
     * @param tag The message's identifying tag.
     * @param comm The communicator this operation applies to.
     * @return The message that has been received.
     */
    template <typename T>
    MPIWCPP17_INLINE std::pair<status_t, payload_out_t<T>> receive(
        size_t count
      , process_t source = process::any
      , tag_t tag = mpiwcpp17::tag::any
      , communicator_t comm = world
    ) {
        auto type = datatype::identify<T>();
        auto out = payload::create_output<T>(count);
        status_t status; guard(MPI_Recv((T*) out, count, type, source, tag, comm, &status));
        return std::pair(status, std::move(out));
    }
}

namespace collective
{
    /**
     * Receives a generic message from a process connected to a communicator.
     * @tparam T The message's contents type.
     * @param source The process to receive the message from.
     * @param tag The message's identifying tag.
     * @param comm The communicator this operation applies to.
     * @return The message that has been received.
     */
    template <typename T>
    MPIWCPP17_INLINE auto receive(
        process_t source = process::any
      , tag_t tag = mpiwcpp17::tag::any
      , communicator_t comm = world
    ) {
        auto status = collective::probe(source, tag, comm);
        auto count  = mpiwcpp17::status::count<T>(status);
        return detail::collective::receive<T>(count, source, tag, comm);
    }
}

/*
 * Exposing the above-defined collective operation into the project's root namespace,
 * allowing it be called with decreased verbosity.
 */
using collective::receive;

MPIWCPP17_END_NAMESPACE
