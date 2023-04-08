/**
 * A thin C++17 wrapper for MPI.
 * @file A fixture structure for a simple geometry point.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpiwcpp17.h>

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

#if defined(MPIWCPP17_AVOID_REFLECTION)

/**
 * Describes the point fixture structure.
 * @return The MPI type descriptor instance.
 */
template <typename T>
inline mpi::datatype::descriptor_t mpi::datatype::describe<point_t<T>>()
{
    return mpi::datatype::descriptor_t(
        &point_t<T>::x
      , &point_t<T>::y
    );
}

#endif
