/**
 * A thin C++17 wrapper for MPI.
 * @file MPI cartesian communicator wrapper.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <array>
#include <cstdint>
#include <utility>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/process.hpp>

#include <mpiwcpp17/detail/topology.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace topology
{
    /**
     * A cartesian topology blueprint responsible for describing the dimensions
     * of a cartesian grid to represent a topology communicator with.
     * @tparam N The topology's number of cartesian dimensions.
     * @see mpiwcpp17::topology::cartesian
     * @since 1.0
     */
    template <size_t N>
    class cartesian_t : public detail::topology::blueprint_t
    {
        private:
            using dimension_t = std::array<int32_t, N>;

        private:
            dimension_t m_dimensions = {};
            dimension_t m_periodic = {};

        public:
            inline constexpr cartesian_t() = default;
            inline cartesian_t(const cartesian_t&) = default;
            inline cartesian_t(cartesian_t&&) = default;

            /**
             * Initializes a new blueprint with cartesian dimensions.
             * @param dimensions The size of each blueprint's cartesian dimensions.
             * @param periodic Informs whether each dimension is periodic or not.
             */
            inline cartesian_t(const dimension_t& dimensions, const dimension_t& periodic = {})
              : m_dimensions (dimensions)
              , m_periodic (periodic)
            {}

            inline cartesian_t& operator=(const cartesian_t&) = default;
            inline cartesian_t& operator=(cartesian_t&&) = default;

        public:
            inline raw_t commit(const raw_t&, bool = true) const override;
            inline static auto extract(const raw_t&) -> cartesian_t;
    };

    /**
     * Commits the described blueprint and creates a new communicator.
     * @tparam N The topology's number of cartesian dimensions.
     * @param comm The original communicator to apply the blueprint to.
     * @param reorder May process ranks be reassigned within new communicator?
     * @return The new topology-applied communicator.
     */
    template <size_t N>
    inline auto cartesian_t<N>::commit(const raw_t& comm, bool reorder) const -> raw_t
    {
        raw_t x;
        const int32_t *dimensions = m_dimensions.data();
        const int32_t *periodic = m_periodic.data();

        guard(MPI_Cart_create(comm, N, dimensions, periodic, reorder, &x));
        return x;
    }

    /**
     * Retrieves the cartesian-topology blueprint applied over the given communicator.
     * @tparam N The topology's number of cartesian dimensions.
     * @param comm The topology-enabled communicator to extract topology from.
     * @return The topology extracted from the given communicator.
     */
    template <size_t N>
    inline auto cartesian_t<N>::extract(const raw_t& comm) -> cartesian_t
    {
        dimension_t dimensions, periodic, coords;
        int32_t *raw_dimensions = dimensions.data();
        int32_t *raw_periodic = periodic.data();
        int32_t *_ = coords.data();

        guard(MPI_Cart_get(comm, N, raw_dimensions, raw_periodic, _));
        return cartesian_t(dimensions, periodic);
    }
}

MPIWCPP17_END_NAMESPACE
