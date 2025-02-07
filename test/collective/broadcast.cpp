/**
 * A thin C++17 wrapper for MPI.
 * @file Test cases for the broadcast collective operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#include <cstdint>
#include <vector>

#include <mpiwcpp17/api.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "resources/point.hpp"

using namespace Catch;

TEST_CASE("broadcast values between processes", "[collective][broadcast]")
{
    auto root = GENERATE(range(0, mpi::global::size));

    /**
     * Tests whether a single scalar value can be seamlessly broadcast to all processes
     * from any process as the root for the operation.
     * @since 1.0
     */
    SECTION("a single scalar value") {
        int value = (root == mpi::global::rank)
            ? (mpi::global::rank + 1) * 2
            : 0;

        int result = mpi::broadcast(&value, 1, root);

        REQUIRE(result == (root + 1) * 2);
    }

    /**
     * Tests whether a container of scalar values can be broadcast to all processes
     * from any process as the root for the operation.
     * @since 1.0
     */
    SECTION("a container of scalar values") {
        constexpr int quantity = 4;
        std::vector<int> value;

        if (root == mpi::global::rank) for (int i = 1; i <= quantity; ++i)
            value.push_back(10 * i + mpi::global::rank);

        auto result = mpi::broadcast(value, root);

        REQUIRE(result.count == quantity);

        for (int i = 0; i < quantity; ++i)
            REQUIRE(result[i] == (10 * (i+1) + root));
    }

    /**
     * Tests whether simple data structures can be broadcast between processes using
     * type reflection or a type description to represent the structure within MPI.
     * @since 1.0
     */
    SECTION("a single default-copyable structure instance") {
        point_t<int> value = (root == mpi::global::rank)
            ? point_t<int> {mpi::global::rank + 1, mpi::global::rank + 2}
            : point_t<int> {0, 0};

        point_t<int> result = mpi::broadcast(&value, 1, root);

        REQUIRE(result.x == root + 1);
        REQUIRE(result.y == root + 2);
    }
}
