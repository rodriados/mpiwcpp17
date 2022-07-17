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

#include <mpiwcpp17/detail/communicator/safety.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail::communicator
{
    /**
     * The basic communicator wrapper. This communicator type simply represents
     * a collection of processes without an intrinsic underlying topology.
     * @since 1.0
     */
    class basic
    {
        public:
            using raw_type = MPI_Comm;

        private:
            raw_type m_comm = MPI_COMM_NULL;

        public:
            const process::rank rank = 0;
            const int32_t size = 0;

        public:
            inline constexpr basic() noexcept = default;

            /**
             * Duplicates the communicator wrapped by another instance.
             * @param other The communicator to be duplicated.
             */
            inline basic(const basic& other)
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
            inline basic(basic&& other) noexcept
              : m_comm {other.m_comm}
              , rank {other.rank}
              , size {other.size}
            {
                new (&other) basic ();
            }

            /**
             * Instantiates a new communicator by acquiring ownership of a raw communicator.
             * @param comm The raw communicator to be acquired identifier.
             */
            inline explicit basic(raw_type comm)
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
            inline ~basic()
            {
                detail::communicator::safety::free(m_comm);
                m_comm = MPI_COMM_NULL;
            }

            /**
             * Frees the resources previously owned and copies another communicator.
             * @param other The communicator to be copied.
             * @return The current communicator instance.
             */
            inline basic& operator=(const basic& other)
            {
                detail::communicator::safety::free(m_comm);
                return *new (this) basic (other);
            }

            /**
             * Frees the resources previously owned and acquires ownership of a communicator.
             * @param other The communicator to have its ownership moved.
             * @return The current communicator instance.
             */
            inline basic& operator=(basic&& other)
            {
                detail::communicator::safety::free(m_comm);
                return *new (this) basic (std::forward<decltype(other)>(other));
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
             * Duplicates the communicator with all its processes and attached information.
             * @return The new duplicated communicator.
             */
            inline auto duplicate() const -> basic
            {
                raw_type x; guard(MPI_Comm_dup(m_comm, &x));
                return basic(x);
            }

            /**
             * Splits processes within the communicator into different channels
             * according to each process's individual selection.
             * @param color The color selected by current process.
             * @param key The key used to assign a process id within the new communicator.
             * @return The communicator obtained from the split.
             */
            inline auto split(int color, process::rank key = process::any) const -> basic
            {
                raw_type x; guard(MPI_Comm_split(m_comm, color, key, &x));
                return basic(x);
            }

            /**
             * Checks whether the wrapper communicator is valid.
             * @return Is the communicator valid?
             */
            inline bool empty() const
            {
                return m_comm == MPI_COMM_NULL;
            }
    };
}

MPIWCPP17_END_NAMESPACE
