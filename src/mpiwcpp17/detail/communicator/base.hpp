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
     * The process communicator wrapper base. This communicator type simply represents
     * a collection of processes without an intrinsic underlying topology.
     * @since 1.0
     */
    class base
    {
        protected:
            using raw_type = MPI_Comm;

        protected:
            const raw_type m_comm = MPI_COMM_NULL;

        public:
            const process::rank rank = 0;
            const int32_t size = 0;

        public:
            inline constexpr base() noexcept = default;

            inline base(const base&);
            inline base(base&&) noexcept;
            inline explicit base(const raw_type&);

            inline base& operator=(const base&);
            inline base& operator=(base&&);
            inline operator const raw_type() const;

            inline ~base();

            inline auto duplicate() const -> base;
            inline auto split(int, process::rank = process::any) const -> base;

            inline bool empty() const;

        protected:
            inline constexpr explicit base(const raw_type&, process::rank, int32_t) noexcept;
    };

    /**
     * Duplicates the communicator wrapped by another instance.
     * @param other The communicator to be duplicated.
     */
    inline base::base(const base& other)
      : rank (other.rank)
      , size (other.size)
    {
        if (!other.empty()) {
            guard(MPI_Comm_dup(other.m_comm, const_cast<raw_type*>(&m_comm)));
        }
    }

    /**
     * Moves and transfers ownership of a wrapped communicator.
     * @param other The communicator to be moved.
     */
    inline base::base(base&& other) noexcept
      : base (other.m_comm, other.rank, other.size)
    {
        new (&other) detail::communicator::base ();
    }

    /**
     * Instantiates a new communicator by acquiring ownership of a raw communicator.
     * @param comm The raw communicator identifier to be acquired.
     */
    inline base::base(const raw_type& comm)
      : m_comm (comm)
    {
        if (!empty()) {
            guard(MPI_Comm_rank(m_comm, const_cast<process::rank*>(&rank)));
            guard(MPI_Comm_size(m_comm, const_cast<int32_t*>(&size)));
            guard(MPI_Comm_set_errhandler(m_comm, MPI_ERRORS_RETURN));
        }
    }

    /**
     * Instantiates a new communicator with its internal properties' values.
     * @param comm The raw communicator identifier to be acquired.
     * @param rank The current process rank within the communicator.
     * @param size The total amount of processes within the communicator.
     */
    inline constexpr base::base(const raw_type& comm, process::rank rank, int32_t size) noexcept
      : m_comm (comm)
      , rank (rank)
      , size (size)
    {}

    /**
     * Frees the resources previously owned and copies another communicator.
     * @param other The communicator to be copied.
     * @return The current communicator instance.
     */
    inline base& base::operator=(const base& other)
    {
        detail::communicator::safety::free(m_comm);
        return *new (this) detail::communicator::base (other);
    }

    /**
     * Frees the resources previously owned and acquires ownership of a communicator.
     * @param other The communicator to have its ownership moved.
     * @return The current communicator instance.
     */
    inline base& base::operator=(base&& other)
    {
        detail::communicator::safety::free(m_comm);
        return *new (this) detail::communicator::base (std::forward<decltype(other)>(other));
    }

    /**
     * Implicit conversion to the raw underlying type. This allows the wrapper
     * to be seamlessly used in native MPI functions.
     * @return The raw communicator identifier.
     */
    inline base::operator const raw_type() const
    {
        return m_comm;
    }

    /**
     * Destroys a communicator wrapper and frees all owned resources.
     * @see base::base
     */
    inline base::~base()
    {
        detail::communicator::safety::free(m_comm);
        m_comm = MPI_COMM_NULL;
    }

    /**
     * Duplicates the communicator with all its processes and attached information.
     * @return The new duplicated communicator.
     */
    inline auto base::duplicate() const -> base
    {
        raw_type c; guard(MPI_Comm_dup(m_comm, &c));
        return detail::communicator::base (c);
    }

    /**
     * Splits processes within the communicator into different communicators according
     * to each process's individual selection.
     * @param color The color selected by current process.
     * @param key The key used to assign a process id within the new communicator.
     * @return The communicator obtained from the split.
     */
    inline auto base::split(int color, process::rank key) const -> base
    {
        raw_type c; guard(MPI_Comm_split(m_comm, color, key, &c));
        return detail::communicator::base (c);
    }

    /**
     * Checks whether the wrapper communicator is valid.
     * @return Is the communicator valid?
     */
    inline bool base::empty() const
    {
        return m_comm == MPI_COMM_NULL;
    }
}

MPIWCPP17_END_NAMESPACE
