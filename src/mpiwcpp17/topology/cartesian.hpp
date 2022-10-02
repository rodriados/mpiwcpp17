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
    class cartesian : public detail::topology::blueprint
    {
        private:
            using dimension_type = std::array<int32_t, N>;

        private:
            dimension_type m_dimensions = {};
            dimension_type m_periodic = {};

        public:
            inline constexpr cartesian() = default;
            inline cartesian(const cartesian&) = default;
            inline cartesian(cartesian&&) = default;

            /**
             * Initializes a new blueprint with cartesian dimensions.
             * @param dimensions The size of each blueprint's cartesian dimensions.
             * @param periodic Informs whether each dimension is periodic or not.
             */
            inline cartesian(const dimension_type& dimensions, const dimension_type& periodic = {})
              : m_dimensions (dimensions)
              , m_periodic (periodic)
            {}

            inline cartesian& operator=(const cartesian&) = default;
            inline cartesian& operator=(cartesian&&) = default;

        public:
            inline raw_type commit(const raw_type&, bool = true) const override;
            inline static auto extract(const raw_type&) -> cartesian;
    };

    /**
     * Commits the described blueprint and creates a new communicator.
     * @tparam N The topology's number of cartesian dimensions.
     * @param comm The original communicator to apply the blueprint to.
     * @param reorder May process ranks be reassigned within new communicator?
     * @return The new topology-applied communicator.
     */
    template <size_t N>
    inline auto cartesian<N>::commit(const raw_type& comm, bool reorder) const -> raw_type
    {
        raw_type x;
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
    inline auto cartesian<N>::extract(const raw_type& comm) -> cartesian
    {
        dimension_type dimensions, periodic, coords;
        int32_t *raw_dimensions = dimensions.data();
        int32_t *raw_periodic = periodic.data();
        int32_t *_ = coords.data();

        guard(MPI_Cart_get(comm, N, raw_dimensions, raw_periodic, _));
        return cartesian(dimensions, periodic);
    }
}

MPIWCPP17_END_NAMESPACE
