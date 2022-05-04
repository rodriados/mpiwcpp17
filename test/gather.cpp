/**
 * A thin C++17 wrapper for MPI.
 * @file Test cases for the gather collective operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#include <cstdint>
#include <vector>

#include <catch.hpp>
#include <mpiwcpp17.h>

/**
 * Test whether a uniform scalar value can be gathered into the a process.
 * @since 1.0
 */
TEST_CASE("can gather uniform scalars into a process", "[global]")
{
    enum { root };
    uint64_t value = (mpi::world.rank + 1) * 2;

    auto result = mpi::gather(&value, 1, root, mpi::flag::uniform());

    if (mpi::world.rank == root) {
        REQUIRE(result.count == mpi::world.size);
        for (int i = 0; i < mpi::world.size; ++i)
            REQUIRE(result[i] == (i + 1) * 2);
    } else {
        REQUIRE(result.count == 0);
    }
}

/**
 * Test whether a varying scalar value can be gathered into a process.
 * @since 1.0
 */
TEST_CASE("can gather varying scalars into a process", "[global]")
{
    enum { root };

    uint32_t base = (mpi::world.rank + 1) * 2;
    uint32_t value[] = { base, base + 1 };
    auto quantity = mpi::world.rank == root ? 1 : 2;

    auto result = mpi::gather(value, quantity, root, mpi::flag::varying());

    if (mpi::world.rank == root) {
        REQUIRE(result.count == (mpi::world.size * 2 - 1));
        REQUIRE(result[0] == 2);
        for (int i = 1; i < mpi::world.size; ++i)
            for (int j = 0; j < 2; ++j)
                REQUIRE(result[i * 2 + j - 1] == ((i + 1) * 2 + j));
    } else {
        REQUIRE(result.count == 0);
    }
}

/**
 * Test whether a varying container can be gathered into a process.
 * @since 1.0
 */
TEST_CASE("can gather varying containers into a process", "[global]")
{
    enum { root };
    std::vector<int16_t> values;

    for (int i = 0; i < mpi::world.rank + 1; ++i)
        values.push_back(i);

    auto result = mpi::gather(values, root);

    if (mpi::world.rank == root) {
        auto size = mpi::world.size;
        REQUIRE(result.count == ((size * (size + 1)) / 2));
        for (int i = 0, k = 0; i < size; ++i)
            for (int j = 0; j <= i; ++j, ++k)
                REQUIRE(result[k] == j);
    } else {
        REQUIRE(result.count == 0);
    }
}
