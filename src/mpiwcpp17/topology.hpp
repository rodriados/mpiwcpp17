/**
 * A thin C++17 wrapper for MPI.
 * @file Helper header for including all MPI communicator topologies.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <utility>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/communicator.hpp>

#include <mpiwcpp17/detail/topology.hpp>
#include <mpiwcpp17/topology/cartesian.hpp>
#include <mpiwcpp17/topology/graph.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace topology
{
    /**
     * The topology communicator. The topology-enabled communicator is able to be
     * used on special neighbor-collective operations.
     * @tparam T The communicator's topology blueprint type.
     * @since 1.0
     */
    template <typename T>
    class communicator_t : public std::enable_if<
        std::is_base_of<detail::topology::blueprint_t, T>::value
      , mpiwcpp17::communicator_t
    >::type
    {
        private:
            using underlying_t = mpiwcpp17::communicator_t;

        protected:
            using raw_t = typename underlying_t::raw_t;

        public:
            inline constexpr communicator_t() noexcept = default;
            inline communicator_t(const communicator_t&) = default;
            inline communicator_t(communicator_t&&) noexcept = default;

            /**
             * Initializes a new topology communicator from a previously created
             * communicator and a topology blueprint to apply over the communicator.
             * @param comm The previously created and base communicator.
             * @param topology The topology blueprint to apply over the communicator.
             * @param reorder May the processes' ranks be reordered in the new communicator?
             */
            inline explicit communicator_t(const raw_t& comm, const T& topology, bool reorder = true)
              : underlying_t (topology.commit(comm, reorder))
            {}

            inline communicator_t& operator=(const communicator_t&) = default;
            inline communicator_t& operator=(communicator_t&&) = default;

            /**
             * Retrieves and extracts the underlying topology blueprint that has
             * been applied over the communicator's processes.
             * @tparam U The topology type to be extracted from communicator.
             * @return The blueprint describing the communicator's topology.
             */
            template <
                typename U = T
              , typename = typename std::enable_if<
                    std::is_base_of<
                        detail::topology::blueprint_t
                      , decltype(U::extract(std::declval<communicator_t>()))
                    >::value
                >::type
            >
            inline auto topology() const
            {
                return U::extract(*this);
            }
    };
}

MPIWCPP17_END_NAMESPACE
