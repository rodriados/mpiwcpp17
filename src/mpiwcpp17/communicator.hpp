/**
 * A thin C++17 wrapper for MPI.
 * @file MPI communicators wrapper and helper functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <cstdint>
#include <utility>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/guard.hpp>

#include <mpiwcpp17/detail/communicator/base.hpp>

MPIWCPP17_BEGIN_NAMESPACE

/**
 * Communicator wrapper. A communicator represents, in some sense, a collection
 * of processes. Each process within a communicator is assigned a rank that uniquely
 * identifies it within such communicator.
 * @since 1.0
 */
class communicator : public detail::communicator::base
{
    protected:
        using base_type = detail::communicator::base;

    public:
        using base_type::base;
        using base_type::operator=;

    //using graph = detail::communicator::graph;
    //template <size_t N = 2> using cartesian = detail::communicator::cartesian<N>;
};

MPIWCPP17_END_NAMESPACE
