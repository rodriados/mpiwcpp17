/**
 * A thin C++17 wrapper for MPI.
 * @file MPI message tag identifiers and global value definitions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace tag
{
    /**
     * The type for identifying a message tag.
     * @since 1.0
     */
    using id = decltype(MPI_ANY_TAG);

    /**
     * Defines a special message tag for any unused or irrelevant tag.
     * @since 1.0
     */
    inline constexpr tag::id any = MPI_ANY_TAG;
}

MPIWCPP17_END_NAMESPACE
