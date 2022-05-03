/**
 * A thin C++17 wrapper for MPI.
 * @file Test cases for the broadcast collective operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#include <cstdint>
#include <vector>

#include <catch.hpp>
#include <mpiwcpp17.h>

/**
 * Tests whether a scalar can be seamlessly broadcast between processes.
 * @since 1.0
 */
TEST_CASE("can broadcast a scalar value", "[global]")
{
    enum { root };
    int32_t value = 0;

    if (mpi::world.rank == root) {
        value = 20;
    }

    int32_t result = mpi::broadcast(&value, 1, root);

    REQUIRE(result == 20);
}

/**
 * Tests whether simple data structures can be broadcast between processes using
 * type reflection to represent the structure automatically within MPI.
 * @since 1.0
 */
TEST_CASE("can broadcast simple data structures with reflection", "[global]")
{
    enum { root };
    struct point { int32_t x, y; };
    struct rectangle { point topright, bottomleft; };

    rectangle value;

    if (mpi::world.rank == root) {
        value.topright = point {10, 15};
        value.bottomleft = point {20, 25};
    }

    rectangle result = mpi::broadcast(value, root);

    REQUIRE(result.topright.x == 10);
    REQUIRE(result.topright.y == 15);
    REQUIRE(result.bottomleft.x == 20);
    REQUIRE(result.bottomleft.y == 25);
}

/**
 * Tests whether a container can be successfully broadcast between processes.
 * @since 1.0
 */
TEST_CASE("can broadcast containers", "[global]")
{
    enum { root };
    std::vector<uint16_t> values;

    if (mpi::world.rank == root) {
        values.push_back(10);
        values.push_back(20);
        values.push_back(30);
        values.push_back(40);
    }

    auto result = mpi::broadcast(values, root);

    REQUIRE(result.count == 4);
    REQUIRE(result[0] == 10);
    REQUIRE(result[1] == 20);
    REQUIRE(result[2] == 30);
    REQUIRE(result[3] == 40);
}
