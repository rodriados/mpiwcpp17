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

using namespace Catch;

SCENARIO("gather values from all processes", "[collective][gather]")
{
    auto root = GENERATE(range(0, mpi::global::size));

    /**
     * Tests whether a single scalar value can be gathered from all processes into
     * any process as the root for the operation.
     * @since 1.0
     */
    GIVEN("a single scalar value") {
        int value = mpi::global::rank + 1;
        auto result = mpi::gather(&value, 1, root, mpi::flag::uniform());

        if (root == mpi::global::rank) THEN("root has all processes' values") {
            REQUIRE(result.count == mpi::global::size);
            for (int i = 0; i < mpi::global::size; ++i)
                REQUIRE(result[i] == (i + 1));
        }

        else THEN("other processes have no values") {
            REQUIRE(result.count == 0);
        }
    }

    /**
     * Tests whether a container with an uniform amount of elements across all processes
     * can be gathered into any process as the root for the operation.
     * @since 1.0
     */
    GIVEN("a uniform container of scalar values") {
        constexpr int quantity = 4;
        std::vector<int> value (quantity);

        for (int i = 0; i < quantity; ++i)
            value[i] = 10 * mpi::global::rank + i;

        auto result = mpi::gather(value, root, mpi::flag::uniform());

        if (root == mpi::global::rank) THEN("root has all processes' values") {
            REQUIRE(result.count == (quantity * mpi::global::size));
            for (int i = 0, k = 0; i < mpi::global::size; ++i)
                for (int j = 0; j < quantity; ++j, ++k)
                    REQUIRE(result[k] == (10 * i + j));
        }

        else THEN("other processes have no values") {
            REQUIRE(result.count == 0);
        }
    }

    /**
     * Tests whether a container with a varying amount of elements between processes
     * can be gathered into any process as the root for the operation.
     * @since 1.0
     */
    GIVEN("a varying container of scalar values") {
        std::vector<int> value (mpi::global::rank + 1);

        for (int i = 0; i <= mpi::global::rank; ++i)
            value[i] = 100 * root + mpi::global::rank * 10 + i;

        auto result = mpi::gather(value, root, mpi::flag::varying());

        if (root == mpi::global::rank) THEN("root has all processes' values") {
            auto size = mpi::global::size;
            REQUIRE(result.count == (size * (size + 1) / 2));
            for (int i = 0, k = 0; i < mpi::global::size; ++i)
                for (int j = 0; j <= i; ++j, ++k)
                    REQUIRE(result[k] == (100 * mpi::global::rank + i * 10 + j));
        }

        else THEN("other processes have no values") {
            REQUIRE(result.count == 0);
        }
    }
}
