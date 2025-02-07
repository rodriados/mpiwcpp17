/**
 * A thin C++17 wrapper for MPI.
 * @file Simple point-like structure for testing.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpiwcpp17/api.h>

/**
 * A simple 2D-geometry point fixture structure.
 * @tparam T The point's dimensions' value type.
 * @since 1.0
 */
template <typename T = double>
struct point_t
{
    T x, y;
};

/**
 * Provides a datatype descriptor identifier for the fixture point structure.
 * @tparam T The point's dimensions' type.
 * @since 3.0
 */
template <typename T>
struct mpi::datatype::provider_t<point_t<T>> {
    inline static mpi::datatype_t provide() {
        return mpi::datatype::provide(
            &point_t<T>::x
          , &point_t<T>::y);
    }
};
