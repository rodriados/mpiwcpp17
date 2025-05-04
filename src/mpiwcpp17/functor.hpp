/**
 * A thin C++17 wrapper for MPI.
 * @file A wrapper for MPI collective operator functors.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/detail/handle.hpp>

MPIWCPP17_BEGIN_NAMESPACE

/**
 * The type for an operator functor instance identifier. An operator identifier
 * is needed for a functor to be used as operator for some collective operations.
 * @since 2.1
 */
struct functor_t : MPIWCPP17_INHERIT_HANDLE(MPI_Op, MPI_Op_free);

namespace functor
{
    /**#@+
     * Registration of readily available MPI operator functors. These operators
     * can be directly used in operations with the types they are built to.
     * @since 1.0
     */
    MPIWCPP17_INLINE const functor_t max     = MPI_MAX;
    MPIWCPP17_INLINE const functor_t min     = MPI_MIN;
    MPIWCPP17_INLINE const functor_t add     = MPI_SUM;
    MPIWCPP17_INLINE const functor_t mul     = MPI_PROD;
    MPIWCPP17_INLINE const functor_t andl    = MPI_LAND;
    MPIWCPP17_INLINE const functor_t andb    = MPI_BAND;
    MPIWCPP17_INLINE const functor_t orl     = MPI_LOR;
    MPIWCPP17_INLINE const functor_t orb     = MPI_BOR;
    MPIWCPP17_INLINE const functor_t xorl    = MPI_LXOR;
    MPIWCPP17_INLINE const functor_t xorb    = MPI_BXOR;
    MPIWCPP17_INLINE const functor_t minloc  = MPI_MINLOC;
    MPIWCPP17_INLINE const functor_t maxloc  = MPI_MAXLOC;
    MPIWCPP17_INLINE const functor_t replace = MPI_REPLACE;
    /**#@-*/
}

MPIWCPP17_END_NAMESPACE
