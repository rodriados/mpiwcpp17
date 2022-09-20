/**
 * A thin C++17 wrapper for MPI.
 * @file Internal helper functions for MPI communicators.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <utility>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/guard.hpp>

MPIWCPP17_BEGIN_NAMESPACE

inline bool finalized();

namespace detail::communicator::safety
{
    /**
     * Verifies whether a communicator can be freed and, if so, frees it.
     * @param target The communicator to be freed if possible.
     */
    inline void free(MPI_Comm target)
    {
        if (target != MPI_COMM_NULL && !mpiwcpp17::finalized()) {
            int compare_world, compare_self;
            guard(MPI_Comm_compare(target, MPI_COMM_WORLD, &compare_world));
            guard(MPI_Comm_compare(target, MPI_COMM_SELF, &compare_self));

            if (compare_world != MPI_IDENT && compare_self != MPI_IDENT) {
                guard(MPI_Comm_free(&target));
            }
        }
    }
}

MPIWCPP17_END_NAMESPACE
