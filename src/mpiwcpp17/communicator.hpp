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
class communicator_t
{
    public:
        using raw_t = MPI_Comm;

    protected:
        std::shared_ptr<void> m_comm;

    public:
        const process_t rank = process::root;
        const int32_t size = 0;

    public:
        inline constexpr communicator_t() noexcept = default;
        inline communicator_t(const communicator_t&) noexcept = default;
        inline communicator_t(communicator_t&&) noexcept = delete;

        inline explicit communicator_t(const raw_t&);

        inline communicator_t& operator=(const communicator_t&) = delete;
        inline communicator_t& operator=(communicator_t&&) = delete;

        /**
         * Implicit conversion to the raw underlying type. This allows the wrapper
         * to be seamlessly used natively by MPI functions.
         * @return The raw communicator identifier.
         */
        inline operator const raw_t() const
        {
            return static_cast<raw_t>(m_comm.get());
        }
};

namespace communicator
{
    /**
     * Duplicates the communicator with all its processes and attached information.
     * @return The new duplicated communicator.
     */
    inline auto duplicate(const communicator_t& comm) -> communicator_t
    {
        communicator_t::raw_t x;
        guard(MPI_Comm_dup(comm, &x));
        return communicator_t (x);
    }

    /**
     * Splits processes within the communicator into different communicators according
     * to each process's individual selection.
     * @param color The color selected by current process.
     * @param key The key used to assign a process id within the new communicator.
     * @return The communicator obtained from the split.
     */
    inline auto split(const communicator_t& comm, int color, process_t key = process::any) -> communicator_t
    {
        communicator_t::raw_t x;
        guard(MPI_Comm_split(comm, color, key, &x));
        return communicator_t (x);
    }

    /**
     * Splits processes within the communicator into different communicators grouping
     * the processes according to their internal types.
     * @param type The type criteria to group processes together.
     * @param key The key used to assign a process id within the new communicator.
     * @return The communicator obtained from the split.
     */
    inline auto split(
        const communicator_t& comm
      , process::type_t type
      , process_t key = process::any
    ) -> communicator_t {
        communicator_t::raw_t x;
        guard(MPI_Comm_split_type(comm, type, key, MPI_INFO_NULL, &x));
        return communicator_t (x);
    }

    /**
     * Checks whether the wrapper communicator is valid.
     * @return Is the communicator valid?
     */
    inline bool empty(const communicator_t& comm)
    {
        return static_cast<communicator_t::raw_t>(comm)
            == static_cast<communicator_t::raw_t>(nullptr);
    }
}

/**
 * Instantiates a new communicator by acquiring ownership of a raw communicator.
 * @param comm The raw communicator identifier to be acquired.
 */
inline communicator_t::communicator_t(const raw_t& comm)
  : m_comm (std::shared_ptr<void>(comm, detail::communicator::safety_free))
{
    guard(MPI_Comm_rank(comm, const_cast<process_t*>(&rank)));
    guard(MPI_Comm_size(comm, const_cast<int32_t*>(&size)));
    guard(MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN));
}

MPIWCPP17_END_NAMESPACE
