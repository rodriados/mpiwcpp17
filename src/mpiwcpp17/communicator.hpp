/**
 * A thin C++17 wrapper for MPI.
 * @file MPI communicators wrapper and helper functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <memory>
#include <cstdint>
#include <utility>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/guard.hpp>

#include <mpiwcpp17/detail/communicator.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace communicator
{
    /**
     * The raw MPI communicator type.
     * @since 3.0
     */
    using raw_t = MPI_Comm;

    /**
     * Communicator wrapper. A communicator represents, in some sense, a collection
     * of processes. Each process within a communicator is assigned a rank that
     * uniquely identifies it within such communicator.
     * @since 1.0
     */
    class wrapper_t
    {
        protected:
            std::shared_ptr<void> m_comm;

        public:
            MPIWCPP17_CONSTEXPR wrapper_t() noexcept = default;

            MPIWCPP17_INLINE wrapper_t(const wrapper_t&) noexcept = default;
            MPIWCPP17_INLINE wrapper_t(wrapper_t&&) noexcept = delete;

            /**
             * Instantiates a new communicator by acquiring ownership of a raw communicator.
             * @param comm The raw communicator identifier to be acquired.
             */
            MPIWCPP17_INLINE explicit wrapper_t(const raw_t& comm)
              : m_comm (std::shared_ptr<void>(static_cast<void*>(comm), detail::communicator::safety_free))
            {
                guard(MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN));
            }

            MPIWCPP17_INLINE wrapper_t& operator=(const wrapper_t&) = delete;
            MPIWCPP17_INLINE wrapper_t& operator=(wrapper_t&&) = delete;

            /**
             * Implicit conversion to the raw underlying type. This allows the wrapper
             * to be seamlessly used natively by MPI functions.
             * @return The raw communicator identifier.
             */
            MPIWCPP17_INLINE operator const raw_t() const
            {
                return static_cast<raw_t>(m_comm.get());
            }
    };

    /**
     * Duplicates the communicator with all its processes and attached information.
     * @return The new duplicated communicator.
     */
    MPIWCPP17_INLINE auto duplicate(const raw_t& comm) -> wrapper_t
    {
        raw_t x; guard(MPI_Comm_dup(comm, &x));
        return wrapper_t (x);
    }

    /**
     * Splits processes within the communicator into different communicators according
     * to each process's individual selection.
     * @param color The color selected by current process.
     * @param key The key used to assign a process id within the new communicator.
     * @return The communicator obtained from the split.
     */
    MPIWCPP17_INLINE auto split(
        const raw_t& comm
      , int color, process_t key = process::any
    ) -> wrapper_t {
        raw_t x; guard(MPI_Comm_split(comm, color, key, &x));
        return wrapper_t (x);
    }

    /**
     * Splits processes within the communicator into different communicators grouping
     * the processes according to their internal types.
     * @param type The type criteria to group processes together.
     * @param key The key used to assign a process id within the new communicator.
     * @return The communicator obtained from the split.
     */
    MPIWCPP17_INLINE auto split(
        const raw_t& comm
      , process::type_t type
      , process_t key = process::any
    ) -> wrapper_t {
        raw_t x; guard(MPI_Comm_split_type(comm, type, key, MPI_INFO_NULL, &x));
        return wrapper_t (x);
    }

    /**
     * Checks whether the wrapper communicator is valid.
     * @return Is the communicator valid?
     */
    MPIWCPP17_INLINE bool empty(const raw_t& comm)
    {
        return comm == static_cast<raw_t>(nullptr);
    }
}

/**
 * Exposing the communicator wrapper to the project's root namespace, allowing it
 * to be referenced by with decreased verbosity.
 * @since 1.0
 */
using communicator_t = communicator::wrapper_t;

/**
 * Informs the rank of the calling process within the given communicator.
 * @param comm The communicator to check the process' rank with.
 * @return The calling process' rank within communicator.
 */
MPIWCPP17_INLINE auto rank(const communicator::raw_t& comm) -> process_t
{
    process_t result; guard(MPI_Comm_rank(comm, &result));
    return result;
}

/**
 * Informs the number of processes within the given communicator.
 * @param comm The communicator to check the number of processes of.
 * @return The number of processes within given communicator.
 */
MPIWCPP17_INLINE auto size(const communicator::raw_t& comm) -> int32_t
{
    int32_t result; guard(MPI_Comm_size(comm, &result));
    return result;
}

MPIWCPP17_END_NAMESPACE
