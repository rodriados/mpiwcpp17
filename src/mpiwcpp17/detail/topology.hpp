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

        protected:
            int32_t m_count = 0;

        public:
            inline constexpr blueprint() = default;
            inline blueprint(const blueprint&) = default;
            inline blueprint(blueprint&&) = default;

            /**
             * Initializes a new blueprint with the number of processes nodes.
             * @param count The number of process nodes of new communicator.
             */
            inline constexpr blueprint(int32_t count)
              : m_count (count)
            {}

            inline blueprint& operator=(const blueprint&) = default;
            inline blueprint& operator=(blueprint&&) = default;

            /**
             * Commits the described blueprint and creates a new communicator.
             * @param comm The original communicator to apply the blueprint to.
             * @param reorder May process ranks be reassigned within new communicator?
             * @return The new topology-applied communicator.
             */
            virtual raw_type commit(const raw_type& comm, bool reorder = true) const = 0;

            /**
             * Retrieves and extracts the topology applied over the communicator.
             * @param comm The communicator to extract the topology from.
             */
            virtual void extract(const raw_type& comm) = 0;
    };
}

MPIWCPP17_END_NAMESPACE
