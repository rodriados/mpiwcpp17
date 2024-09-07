/**
 * A thin C++17 wrapper for MPI.
 * @file A wrapper for MPI collective operator functors.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>
#include <utility>

#include <mpiwcpp17/environment.h>

MPIWCPP17_BEGIN_NAMESPACE

/**
 * The type for an operator functor instance identifier. An operator identifier
 * is needed for a functor to be used as operator for some collective operations.
 * @since 3.0
 */
using functor_t = MPI_Op;

MPIWCPP17_END_NAMESPACE

#include <mpiwcpp17/detail/functor.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace functor
{
    /**#@+
     * Registration of readily available MPI operator functors. These operators
     * can be directly used in operations with the types they are built to.
     * @since 1.0
     */
    MPIWCPP17_CONSTEXPR const functor_t max     = MPI_MAX;
    MPIWCPP17_CONSTEXPR const functor_t min     = MPI_MIN;
    MPIWCPP17_CONSTEXPR const functor_t add     = MPI_SUM;
    MPIWCPP17_CONSTEXPR const functor_t mul     = MPI_PROD;
    MPIWCPP17_CONSTEXPR const functor_t andl    = MPI_LAND;
    MPIWCPP17_CONSTEXPR const functor_t andb    = MPI_BAND;
    MPIWCPP17_CONSTEXPR const functor_t orl     = MPI_LOR;
    MPIWCPP17_CONSTEXPR const functor_t orb     = MPI_BOR;
    MPIWCPP17_CONSTEXPR const functor_t xorl    = MPI_LXOR;
    MPIWCPP17_CONSTEXPR const functor_t xorb    = MPI_BXOR;
    MPIWCPP17_CONSTEXPR const functor_t minloc  = MPI_MINLOC;
    MPIWCPP17_CONSTEXPR const functor_t maxloc  = MPI_MAXLOC;
    MPIWCPP17_CONSTEXPR const functor_t replace = MPI_REPLACE;
    /**#@-*/
}

MPIWCPP17_END_NAMESPACE
