/**
 * A thin C++17 wrapper for MPI.
 * @file MPI communicators wrapper and helper functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>
#include <cstdint>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/global.hpp>
#include <mpiwcpp17/guard.hpp>

#include <mpiwcpp17/detail/attribute.hpp>
#include <mpiwcpp17/detail/raii.hpp>

MPIWCPP17_BEGIN_NAMESPACE

/**
 * The raw MPI communicator identifier type.
 * This type is used to identify groups of processes that may participate in collective
 * operations together. Is is possible to create subgroups to allow processes to
 * perform operations with subsets of the global group of processes.
 * @since 2.1
 */
using communicator_t = MPI_Comm;

namespace communicator
{
    /**
     * Declares communicator attribute namespace and corresponding functions.
     * Attributes are identified by keys that can be used to attach and retrieve
     * generic data from communicators.
     * @since 2.1
     */
    MPIWCPP17_ATTRIBUTE_DECLARE(
        communicator_t
      , MPI_Comm_create_keyval, MPI_Comm_free_keyval
      , MPI_Comm_get_attr, MPI_Comm_set_attr, MPI_Comm_delete_attr
      , MPI_COMM_DUP_FN, MPI_COMM_NULL_DELETE_FN
    )

    /**
     * Informs the rank of the calling process within the given communicator.
     * @param comm The communicator to check the process' rank with.
     * @return The calling process' rank within communicator.
     */
    MPIWCPP17_INLINE process_t rank(communicator_t comm)
    {
        process_t rank;
        guard(MPI_Comm_rank(comm, &rank));
        return rank;
    }

    /**
     * Informs the number of processes within the given communicator.
     * @param comm The communicator to check the number of processes of.
     * @return The number of processes within given communicator.
     */
    MPIWCPP17_INLINE int32_t size(communicator_t comm)
    {
        int32_t nproc;
        guard(MPI_Comm_size(comm, &nproc));
        return nproc;
    }

    /**
     * Duplicates the communicator with all its processes and attached information.
     * @param comm The communicator to be duplicated.
     * @return The new duplicated communicator.
     */
    MPIWCPP17_INLINE communicator_t duplicate(communicator_t comm)
    {
        communicator_t c;
        guard(MPI_Comm_dup(comm, &c));
        return detail::raii_t::attach(c, &MPI_Comm_free);
    }

    /**
     * Splits processes within the communicator into different communicators according
     * to each process's individual selection.
     * @param comm The communicator to be split.
     * @param color The color selected by current process.
     * @param key The key used to assign a process id within the new communicator.
     * @return The communicator obtained from the split.
     */
    MPIWCPP17_INLINE communicator_t split(
        communicator_t comm
      , int color, process_t key = process::any
    ) {
        communicator_t c;
        guard(MPI_Comm_split(comm, color, key, &c));
        return detail::raii_t::attach(c, &MPI_Comm_free);
    }

    /**
     * Splits processes within the communicator into different communicators grouping
     * the processes according to their internal types.
     * @param comm The communicator to be split.
     * @param type The type criteria to group processes together.
     * @param key The key used to assign a process id within the new communicator.
     * @return The communicator obtained from the split.
     */
    MPIWCPP17_INLINE communicator_t split(
        communicator_t comm
      , process::type_t type
      , process_t key = process::any
    ) {
        communicator_t c;
        guard(MPI_Comm_split_type(comm, type, key, MPI_INFO_NULL, &c));
        return detail::raii_t::attach(c, &MPI_Comm_free);
    }

    /**
     * Checks if the given communicator is in a valid state.
     * @param comm The communicator to check if in a valid state.
     * @return Is the communicator valid?
     */
    MPIWCPP17_INLINE bool empty(communicator_t comm)
    {
        return comm == MPI_COMM_NULL;
    }

    /**
     * Frees communicator resources after checking if it is not MPI-internal.
     * @param comm The communicator to be freed if possible.
     */
    MPIWCPP17_INLINE void free(communicator_t comm)
    {
        if (!communicator::empty(comm) && !finalized())
            if (comm != MPI_COMM_WORLD && comm != MPI_COMM_SELF)
                if (!detail::raii_t::detach(comm))
                    guard(MPI_Comm_free(&comm));
    }
}

MPIWCPP17_END_NAMESPACE
