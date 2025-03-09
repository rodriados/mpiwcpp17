/**
 * A thin C++17 wrapper for MPI.
 * @file MPI generic key-value information wrapper.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2025-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.h>

MPIWCPP17_BEGIN_NAMESPACE

/**
 * The raw MPI key-value information identifier type.
 * This type is used to reference an internal MPI key-value mapping for generic
 * information supported by some MPI functions.
 * @since 2.1
 */
using info_t = MPI_Info;

namespace info
{
    /**
     * The invalid or empty key-value information instance.
     * This can be used to ignore information when required by an MPI call or to
     * denote that the mapping is currently empty or in an invalid state.
     * @since 2.1
     */
    MPIWCPP17_CONSTEXPR const info_t null = MPI_INFO_NULL;
}

MPIWCPP17_END_NAMESPACE
