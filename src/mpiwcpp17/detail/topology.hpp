/**
 * A thin C++17 wrapper for MPI.
 * @file Internal helper functions and classes for topology.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <cstdint>
#include <utility>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/communicator.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail::topology
{
    /**
     * The abstract base for all topology-enabled communicator blueprints.
     * @since 1.0
     */
    class blueprint_t
    {
        public:
            using raw_t = mpiwcpp17::communicator_t::raw_t;

        public:
            /**
             * Commits the described blueprint and creates a new communicator.
             * @param comm The original communicator to apply the blueprint to.
             * @param reorder May process ranks be reassigned within new communicator?
             * @return The new topology-applied communicator.
             */
            virtual raw_t commit(const raw_t& comm, bool reorder = true) const = 0;
    };
}

MPIWCPP17_END_NAMESPACE
