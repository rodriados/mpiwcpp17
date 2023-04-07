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
struct communicator_t
{
    public:
        using raw_t = MPI_Comm;

    protected:
        raw_t m_comm = MPI_COMM_NULL;

    public:
        const process_t rank = process::root;
        const int32_t size = 0;

    public:
        inline constexpr communicator_t() noexcept = default;

        inline communicator_t(const communicator_t&);
        inline communicator_t(communicator_t&&) noexcept;
        inline explicit communicator_t(const raw_t&);

        inline communicator_t& operator=(const communicator_t&);
        inline communicator_t& operator=(communicator_t&&);
        inline operator const raw_t() const;

        inline ~communicator_t();

        inline auto duplicate() const -> communicator_t;
        inline auto split(int, process_t = process::any) const -> communicator_t;

        inline bool empty() const;

    protected:
        inline constexpr explicit communicator_t(const raw_t&, process_t, int32_t) noexcept;
};

/**
 * Duplicates the communicator wrapped by another instance.
 * @param other The communicator to be duplicated.
 */
inline communicator_t::communicator_t(const communicator_t& other)
  : rank (other.rank)
  , size (other.size)
{
    if (!other.empty()) {
        guard(MPI_Comm_dup(other.m_comm, const_cast<raw_t*>(&m_comm)));
    }
}

/**
 * Moves and transfers ownership of a wrapped communicator.
 * @param other The communicator to be moved.
 */
inline communicator_t::communicator_t(communicator_t&& other) noexcept
  : communicator_t (other.m_comm, other.rank, other.size)
{
    new (&other) communicator_t ();
}

/**
 * Instantiates a new communicator by acquiring ownership of a raw communicator.
 * @param comm The raw communicator identifier to be acquired.
 */
inline communicator_t::communicator_t(const raw_t& comm)
  : m_comm (comm)
{
    if (!empty()) {
        guard(MPI_Comm_rank(m_comm, const_cast<process_t*>(&rank)));
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
inline constexpr communicator_t::communicator_t(const raw_t& comm, process_t rank, int32_t size) noexcept
  : m_comm (comm)
  , rank (rank)
  , size (size)
{}

/**
 * Frees the resources previously owned and copies another communicator.
 * @param other The communicator to be copied.
 * @return The current communicator instance.
 */
inline communicator_t& communicator_t::operator=(const communicator_t& other)
{
    if (m_comm == other.m_comm) { return *this; }
    detail::communicator::safety_free(m_comm);
    return *new (this) communicator_t (other);
}

/**
 * Frees the resources previously owned and acquires ownership of a communicator.
 * @param other The communicator to have its ownership moved.
 * @return The current communicator instance.
 */
inline communicator_t& communicator_t::operator=(communicator_t&& other)
{
    if (m_comm == other.m_comm) { return *this; }
    detail::communicator::safety_free(m_comm);
    return *new (this) communicator_t (std::forward<decltype(other)>(other));
}

/**
 * Implicit conversion to the raw underlying type. This allows the wrapper
 * to be seamlessly used in native MPI functions.
 * @return The raw communicator identifier.
 */
inline communicator_t::operator const raw_t() const
{
    return m_comm;
}

/**
 * Destroys a communicator wrapper and frees all owned resources.
 * @see communicator::communicator
 */
inline communicator_t::~communicator_t()
{
    detail::communicator::safety_free(m_comm);
    m_comm = MPI_COMM_NULL;
}

/**
 * Duplicates the communicator with all its processes and attached information.
 * @return The new duplicated communicator.
 */
inline auto communicator_t::duplicate() const -> communicator_t
{
    raw_t x; guard(MPI_Comm_dup(m_comm, &x));
    return communicator_t (x);
}

/**
 * Splits processes within the communicator into different communicators according
 * to each process's individual selection.
 * @param color The color selected by current process.
 * @param key The key used to assign a process id within the new communicator.
 * @return The communicator obtained from the split.
 */
inline auto communicator_t::split(int color, process_t key) const -> communicator_t
{
    raw_t x; guard(MPI_Comm_split(m_comm, color, key, &x));
    return communicator_t (x);
}

/**
 * Checks whether the wrapper communicator is valid.
 * @return Is the communicator valid?
 */
inline bool communicator_t::empty() const
{
    return m_comm == MPI_COMM_NULL;
}

MPIWCPP17_END_NAMESPACE
