/**
 * A thin C++17 wrapper for MPI.
 * @file MPI message tag identifiers and global value definitions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.h>

MPIWCPP17_BEGIN_NAMESPACE

namespace tag
{
    /**
     * The identification or disambiguation value of a reception operation.
     * @since 3.0
     */
    using raw_t = decltype(MPI_ANY_TAG);

    enum : raw_t
    {
        /**
         * The special tag value for denoting that tags are unused or irrelevant for
         * the reception operation.
         * @since 1.0
         */
        any = MPI_ANY_TAG

        /**
         * The special tag value for denoting the highest possible tag value.
         * @since 2.0
         */
      , ub = MPI_TAG_UB
    };
}

/**
 * Exposing the tag type to the project's root namespace, allowing it to be referenced
 * by with decreased verbosity.
 * @since 1.0
 */
using tag_t = tag::raw_t;

MPIWCPP17_END_NAMESPACE
