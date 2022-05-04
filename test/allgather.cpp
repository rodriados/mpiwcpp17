/**
 * A thin C++17 wrapper for MPI.
 * @file Test cases for the all-gather collective operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#include <cstdint>
#include <vector>

#include <catch.hpp>
#include <mpiwcpp17.h>

/**
 * Test whether a uniform scalar value can be gathered into all processes.
 * @since 1.0
 */
TEST_CASE("can gather uniform scalars to all processes", "[global]")
{
    enum { root };
    uint32_t value = (mpi::world.rank + 1) * 2;

    auto result = mpi::allgather(&value, 1, mpi::flag::uniform());

    REQUIRE(result.count == mpi::world.size);
    for (int i = 0; i < mpi::world.size; ++i)
        REQUIRE(result[i] == (i + 1) * 2);
}

/**
 * Test whether a varying scalar value can be gathered into all processes.
 * @since 1.0
 */
TEST_CASE("can gather varying scalars to all processes", "[global]")
{
    enum { root };

    uint32_t base = (mpi::world.rank + 1) * 2;
    uint32_t value[] = { base, base + 1 };
    auto quantity = mpi::world.rank == root ? 1 : 2;

    auto result = mpi::allgather(value, quantity, mpi::flag::varying());

    REQUIRE(result.count == (mpi::world.size * 2 - 1));
    REQUIRE(result[0] == 2);
    for (int i = 1, k = 1; i < mpi::world.size; ++i)
        for (int j = 0; j < 2; ++j, ++k)
            REQUIRE(result[k] == ((i + 1) * 2 + j));
}

/**
 * Test whether a varying container can be gathered into all processes.
 * @since 1.0
 */
TEST_CASE("can gather varying containers to all processes", "[global]")
{
    enum { root };
    std::vector<int16_t> values;

    for (int i = 0; i < mpi::world.rank + 1; ++i)
        values.push_back(i);

    auto result = mpi::allgather(values);

    auto size = mpi::world.size;
    REQUIRE(result.count == ((size * (size + 1)) / 2));
    for (int i = 0, k = 0; i < size; ++i)
        for (int j = 0; j <= i; ++j, ++k)
            REQUIRE(result[k] == j);
}
