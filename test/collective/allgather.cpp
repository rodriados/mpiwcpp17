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

SCENARIO("gather values into all processes", "[collective][allgather]")
{
    /**
     * Tests whether a single scalar value can be gathered into all processes.
     * @since 1.0
     */
    GIVEN("a single scalar value") {
        int value = mpi::global::rank + 1;
        auto result = mpi::allgather(&value, 1, mpi::world, mpi::flag::uniform());

        THEN("all processes have all values") {
            REQUIRE(result.count == mpi::global::size);
            for (int i = 0; i < mpi::global::size; ++i)
                REQUIRE(result[i] == (i + 1));
        }
    }

    /**
     * Tests whether a container with an uniform amount of elements across all processes
     * can be gathered into all processes.
     * @since 1.0
     */
    GIVEN("a uniform container of scalar values") {
        constexpr int quantity = 4;
        std::vector<int> value (quantity);

        for (int i = 0; i < quantity; ++i)
            value[i] = 10 * mpi::global::rank + i;

        auto result = mpi::allgather(value, mpi::world, mpi::flag::uniform());

        THEN("all processes have all values") {
            REQUIRE(result.count == (quantity * mpi::global::size));
            for (int i = 0, k = 0; i < mpi::global::size; ++i)
                for (int j = 0; j < quantity; ++j, ++k)
                    REQUIRE(result[k] == (10 * i + j));
        }
    }

    /**
     * Tests whether a container with a varying amount of elements between processes
     * can be gathered into all processes.
     * @since 1.0
     */
    GIVEN("a varying container of scalar values") {
        std::vector<int> value (mpi::global::rank + 1);

        for (int i = 0; i <= mpi::global::rank; ++i)
            value[i] = mpi::global::rank * 10 + i;

        auto result = mpi::allgather(value, mpi::world, mpi::flag::varying());

        THEN("all processes have all values") {
            auto size = mpi::global::size;
            REQUIRE(result.count == (size * (size + 1) / 2));
            for (int i = 0, k = 0; i < mpi::global::size; ++i)
                for (int j = 0; j <= i; ++j, ++k)
                    REQUIRE(result[k] == (i * 10 + j));
        }
    }
}
