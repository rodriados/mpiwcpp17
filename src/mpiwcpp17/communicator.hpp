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

MPIWCPP17_BEGIN_NAMESPACE

MPIWCPP17_INLINE bool finalized();

/**
 * The raw MPI communicator reference type.
 * @since 3.0
 */
using communicator_t = MPI_Comm;

/**
 * Informs the rank of the calling process within the given communicator.
 * @param comm The communicator to check the process' rank with.
 * @return The calling process' rank within communicator.
 */
MPIWCPP17_INLINE process_t rank(const communicator_t& comm)
{
    process_t rank; guard(MPI_Comm_rank(comm, &rank));
    return rank;
}

/**
 * Informs the number of processes within the given communicator.
 * @param comm The communicator to check the number of processes of.
 * @return The number of processes within given communicator.
 */
MPIWCPP17_INLINE int32_t size(const communicator_t& comm)
{
    int32_t size; guard(MPI_Comm_size(comm, &size));
    return size;
}

namespace communicator
{
    /**
     * Duplicates the communicator with all its processes and attached information.
     * @param comm The communicator to be duplicated.
     * @return The new duplicated communicator.
     */
    MPIWCPP17_INLINE communicator_t duplicate(const communicator_t& comm)
    {
        communicator_t nc; guard(MPI_Comm_dup(comm, &nc));
        return nc;
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
        const communicator_t& comm
      , int color, process_t key = process::any
    ) {
        communicator_t nc; guard(MPI_Comm_split(comm, color, key, &nc));
        return nc;
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
        const communicator_t& comm
      , process::type_t type
      , process_t key = process::any
    ) {
        communicator_t nc; guard(MPI_Comm_split_type(comm, type, key, MPI_INFO_NULL, &nc));
        return nc;
    }

    /**
     * Checks whether the wrapper communicator is valid.
     * @param comm The communicator to check if empty.
     * @return Is the communicator valid?
     */
    MPIWCPP17_INLINE bool empty(const communicator_t& comm)
    {
        return comm == MPI_COMM_NULL;
    }

    /**
     * Verifies whether a communicator can be freed and, if so, frees it.
     * @param comm The communicator to be freed if possible.
     */
    MPIWCPP17_INLINE void free(communicator_t& comm)
    {
        if (!empty(comm) && !mpiwcpp17::finalized()) {
            int compare_world, compare_self;
            guard(MPI_Comm_compare(comm, MPI_COMM_WORLD, &compare_world));
            guard(MPI_Comm_compare(comm, MPI_COMM_SELF, &compare_self));
            if (compare_world != MPI_IDENT && compare_self != MPI_IDENT) {
                guard(MPI_Comm_free(&comm));
            }
        }
    }
}

MPIWCPP17_END_NAMESPACE
