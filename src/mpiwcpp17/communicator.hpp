/**
 * A thin C++17 wrapper for MPI.
 * @file MPI communicators wrapper and helper functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <cstdint>
#include <utility>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/guard.hpp>

#include <mpiwcpp17/detail/communicator.hpp>

MPIWCPP17_BEGIN_NAMESPACE

/**
 * Communicator wrapper. A communicator represents, in some sense, a collection
 * of processes. Each process within a communicator is assigned a rank that uniquely
 * identifies it within such communicator.
 * @since 1.0
 */
class communicator
{
    public:
        using raw_type = MPI_Comm;

    private:
        raw_type m_comm = MPI_COMM_NULL;

    public:
        const process::rank rank = 0;
        const int32_t size = 0;

    public:
        inline constexpr communicator() noexcept = default;

        /**
         * Duplicates the communicator wrapped by another instance.
         * @param other The communicator to be duplicated.
         */
        inline communicator(const communicator& other)
          : rank {other.rank}
          , size {other.size}
        {
            if (!other.empty()) {
                guard(MPI_Comm_dup(other.m_comm, &m_comm));
            }
        }

        /**
         * Moves and transfers ownership of a wrapped communicator.
         * @param other The communicator to be moved.
         */
        inline communicator(communicator&& other) noexcept
          : m_comm {other.m_comm}
          , rank {other.rank}
          , size {other.size}
        {
            new (&other) communicator ();
        }

        /**
         * Instantiates a new communicator by acquiring ownership of a raw communicator.
         * @param comm The raw communicator to be acquired identifier.
         */
        inline explicit communicator(raw_type comm)
          : m_comm {comm}
        {
            if (!empty()) {
                guard(MPI_Comm_rank(m_comm, const_cast<process::rank*>(&rank)));
                guard(MPI_Comm_size(m_comm, const_cast<int32_t*>(&size)));
                guard(MPI_Comm_set_errhandler(m_comm, MPI_ERRORS_RETURN));
            }
        }

        /**
         * Destroys a communicator wrapper and frees all owned resources.
         * @see communicator::communicator
         */
        inline ~communicator()
        {
            detail::communicator::safe_free(m_comm);
            m_comm = MPI_COMM_NULL;
        }

        /**
         * Frees the resources previously owned and copies another communicator.
         * @param other The communicator to be copied.
         * @return The current communicator instance.
         */
        inline communicator& operator=(const communicator& other)
        {
            detail::communicator::safe_free(m_comm);
            return *new (this) communicator (other);
        }

        /**
         * Frees the resources previously owned and acquires ownership of a communicator.
         * @param other The communicator to have its ownership moved.
         * @return The current communicator instance.
         */
        inline communicator& operator=(communicator&& other)
        {
            detail::communicator::safe_free(m_comm);
            return *new (this) communicator (std::forward<decltype(other)>(other));
        }

        /**
         * Implicit conversion to the raw underlying type. This allows the wrapper
         * to be seamlessly used in native MPI functions.
         * @return The raw communicator identifier.
         */
        inline operator const raw_type() const
        {
            return m_comm;
        }

        /**
         * Checks whether the wrapper communicator is valid.
         * @return Is the communicator valid?
         */
        inline bool empty() const
        {
            return m_comm == MPI_COMM_NULL;
        }

    public:
        /**
         * Duplicates a communicator with all its processes and attached information.
         * @param comm The communicator to be duplicated.
         * @return The new duplicated communicator.
         */
        inline static communicator duplicate(const communicator& comm)
        {
            raw_type new_comm;
            guard(MPI_Comm_dup(comm.m_comm, &new_comm));
            return communicator(new_comm);
        }

        /**
         * Splits processes within the communicator into different channels according
         * to each process's individual selection.
         * @param comm The communicator to be split.
         * @param color The color selected by current process.
         * @param key The key used to assign a process id within the new communicator.
         * @return The communicator obtained from the split.
         */
        inline static communicator split(const communicator& comm, int color, process::rank key = process::any)
        {
            raw_type new_comm;
            guard(MPI_Comm_split(comm.m_comm, color, key, &new_comm));
            return communicator(new_comm);
        }
};

MPIWCPP17_END_NAMESPACE
