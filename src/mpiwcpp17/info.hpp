/**
 * A thin C++17 wrapper for MPI.
 * @file MPI generic key-value information wrapper.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2025-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/detail/handle.hpp>

MPIWCPP17_BEGIN_NAMESPACE

/**
 * The raw MPI key-value information identifier type.
 * This type is used to reference an internal MPI key-value mapping for generic
 * information supported by some MPI functions.
 * @since 2.1
 */
struct info_t : MPIWCPP17_INHERIT_HANDLE(MPI_Info, MPI_Info_free);

namespace info
{
    /**
     * The invalid or empty key-value information instance.
     * This can be used to ignore information when required by an MPI call or to
     * denote that the mapping is currently empty or in an invalid state.
     * @since 2.1
     */
    MPIWCPP17_INLINE const info_t null = MPI_INFO_NULL;
}

MPIWCPP17_END_NAMESPACE
