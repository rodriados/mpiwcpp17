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
#include <mpiwcpp17/guard.hpp>

#include <mpiwcpp17/detail/tracker.hpp>
#include <mpiwcpp17/detail/attribute.hpp>

MPIWCPP17_BEGIN_NAMESPACE
MPIWCPP17_FWD_GLOBAL_STATUS_FUNCTIONS

/**
 * The raw MPI communicator reference type.
 * @since 3.0
 */
using communicator_t = MPI_Comm;

namespace communicator
{
    /**
     * Declares communicator attribute namespace and corresponding functions.
     * @since 3.0
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
        process_t rank; guard(MPI_Comm_rank(comm, &rank));
        return rank;
    }

    /**
     * Informs the number of processes within the given communicator.
     * @param comm The communicator to check the number of processes of.
     * @return The number of processes within given communicator.
     */
    MPIWCPP17_INLINE int32_t size(communicator_t comm)
    {
        int32_t size; guard(MPI_Comm_size(comm, &size));
        return size;
    }

    /**
     * Duplicates the communicator with all its processes and attached information.
     * @param comm The communicator to be duplicated.
     * @return The new duplicated communicator.
     */
    MPIWCPP17_INLINE communicator_t duplicate(communicator_t comm)
    {
        communicator_t dup; guard(MPI_Comm_dup(comm, &dup));
        return detail::tracker_t::add(dup, &MPI_Comm_free);
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
        communicator_t newcomm; guard(MPI_Comm_split(comm, color, key, &newcomm));
        return detail::tracker_t::add(newcomm, &MPI_Comm_free);
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
        communicator_t newcomm; guard(MPI_Comm_split_type(comm, type, key, MPI_INFO_NULL, &newcomm));
        return detail::tracker_t::add(newcomm, &MPI_Comm_free);
    }

    /**
     * Checks whether the wrapper communicator is valid.
     * @param comm The communicator to check if empty.
     * @return Is the communicator valid?
     */
    MPIWCPP17_INLINE bool empty(communicator_t comm)
    {
        return comm == MPI_COMM_NULL;
    }

    /**
     * Verifies whether a communicator can be freed and, if so, frees it.
     * @param comm The communicator to be freed if possible.
     */
    MPIWCPP17_INLINE void free(communicator_t comm)
    {
        if (!empty(comm) && !finalized()) {
            int compare_world, compare_self;
            guard(MPI_Comm_compare(comm, MPI_COMM_WORLD, &compare_world));
            guard(MPI_Comm_compare(comm, MPI_COMM_SELF, &compare_self));
            if (compare_world != MPI_IDENT && compare_self != MPI_IDENT) {
                if (!detail::tracker_t::remove(comm))
                    guard(MPI_Comm_free(&comm));
            }
        }
    }
}

MPIWCPP17_END_NAMESPACE
