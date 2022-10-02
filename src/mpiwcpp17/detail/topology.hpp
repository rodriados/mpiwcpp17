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
    class blueprint
    {
        public:
            using raw_type = mpiwcpp17::communicator::raw_type;

        public:
            /**
             * Commits the described blueprint and creates a new communicator.
             * @param comm The original communicator to apply the blueprint to.
             * @param reorder May process ranks be reassigned within new communicator?
             * @return The new topology-applied communicator.
             */
            virtual raw_type commit(const raw_type& comm, bool reorder = true) const = 0;
    };
}

MPIWCPP17_END_NAMESPACE
