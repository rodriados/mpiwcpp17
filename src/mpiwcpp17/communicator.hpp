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
#include <mpiwcpp17/info.hpp>

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

/*
 * Auxiliary macros for implementing functions that wrap the creation of new communicators.
 * The newly created communicators are automatically attached to RAII.
 * @param x The communicator to be attached to RAII.
 * @param B The call block to be wrapped.
 */
#define MPIWCPP17_COMM_RAII(x)  detail::raii_t::attach(x, &MPI_Comm_free)
#define MPIWCPP17_COMM_CALL(B)  MPIWCPP17_COMM_RAII(MPIWCPP17_GUARD_CALL(communicator_t, B))

namespace communicator
{
    /**
     * The global world communicator instance.
     * This communicator contains the original processes that were spawned by MPI
     * when initializing its execution and cannot be modified in any way.
     * @since 2.1
     */
    MPIWCPP17_CONSTEXPR const communicator_t world = mpiwcpp17::world;

    /**
     * The invalid or empty communicator instance.
     * This can be used to verify whether a communicator is not in a valid state
     * or to denote an empty communicator.
     * @since 2.1
     */
    MPIWCPP17_CONSTEXPR const communicator_t null = MPI_COMM_NULL;

    /**
     * The communicator containing only the calling process.
     * This communicator is conformed of only the calling process, thus it does
     * not enable message communication between different processes.
     * @since 2.1
     */
    MPIWCPP17_CONSTEXPR const communicator_t self = MPI_COMM_SELF;

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
     * The comparison likeness between communicators.
     * The likeness between communicators are determined by which processes are
     * connected to each communicator and their corresponding order.
     * @since 2.1
     */
    enum likeness_t : int
    {
        /**
         * Communicators are identical when the processes connected to them are
         * the same while keeping the identical order in both.
         * @since 2.1
         */
        identical = MPI_IDENT

        /**
         * Communicators are considered congruent when their underlying groups are
         * identical in constituents and order but differ by context.
         * @since 2.1
         */
      , congruent = MPI_CONGRUENT

        /**
         * Communicators are similar when the set of processes contained by both
         * are the same but they differ in order.
         * @since 2.1
         */
      , similar   = MPI_SIMILAR

        /**#@+
         * Communicators are considered unequal or different when the sets of processes
         * contained by each communicator differ from one another.
         * @since 2.1
         */
      , unequal   = MPI_UNEQUAL
      , different = unequal
        /**#@-*/
    };

    /**
     * Informs the rank of the calling process within the given communicator.
     * @param comm The communicator to check the process' rank with.
     * @return The calling process' rank within communicator.
     */
    MPIWCPP17_INLINE process_t rank(communicator_t comm)
    {
        return MPIWCPP17_GUARD_CALL(process_t, MPI_Comm_rank(comm, &_));
    }

    /**
     * Informs the number of processes within the given communicator.
     * @param comm The communicator to check the number of processes of.
     * @return The number of processes within given communicator.
     */
    MPIWCPP17_INLINE int32_t size(communicator_t comm)
    {
        return MPIWCPP17_GUARD_CALL(int32_t, MPI_Comm_size(comm, &_));
    }

    /**
     * Duplicates the communicator with all its processes and attached information.
     * @param comm The communicator to be duplicated.
     * @return The new duplicated communicator.
     */
    MPIWCPP17_INLINE communicator_t duplicate(communicator_t comm)
    {
        return MPIWCPP17_COMM_CALL(MPI_Comm_dup(comm, &_));
    }

    /**
     * Duplicates the communicator with new key-value information.
     * @param comm The communicator to be duplicated.
     * @param info The key-value information to attach to new communicator.
     * @return The new duplicated communicator.
     */
    MPIWCPP17_INLINE communicator_t duplicate(communicator_t comm, info_t info)
    {
        return MPIWCPP17_COMM_CALL(MPI_Comm_dup_with_info(comm, info, &_));
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
        return MPIWCPP17_COMM_CALL(MPI_Comm_split(comm, color, key, &_));
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
      , info_t info = info::null
    ) {
        return MPIWCPP17_COMM_CALL(MPI_Comm_split_type(comm, type, key, info, &_));
    }

    /**
     * Compares two communicators and informs their likeness to one another.
     * @param a The first communicator to be compared.
     * @param b The second communicator to be compared.
     * @return The likeness between the communicators.
     */
    MPIWCPP17_INLINE likeness_t compare(communicator_t a, communicator_t b)
    {
        return (likeness_t) MPIWCPP17_GUARD_CALL(int, MPI_Comm_compare(a, b, &_));
    }

    /**
     * Checks if the given communicator is in a valid state.
     * @param comm The communicator to check if in a valid state.
     * @return Is the communicator valid and not empty?
     */
    MPIWCPP17_INLINE bool empty(communicator_t comm)
    {
        return comm == communicator::null;
    }

    /**
     * Frees communicator resources after checking if it is not MPI-internal.
     * @param comm The communicator to be freed if possible.
     */
    MPIWCPP17_INLINE void free(communicator_t comm)
    {
        if (!communicator::empty(comm) && !finalized())
            if (comm != communicator::world && comm != communicator::self)
                if (!detail::raii_t::detach(comm))
                    guard(MPI_Comm_free(&comm));
    }
}

#undef MPIWCPP17_COMM_CALL
#undef MPIWCPP17_COMM_RAII

MPIWCPP17_END_NAMESPACE
