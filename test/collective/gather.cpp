/**
 * A thin C++17 wrapper for MPI.
 * @file Test cases for the gather collective operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#include <cstdint>
#include <vector>

#include <mpiwcpp17/api.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>

using namespace Catch;

TEST_CASE("gather values from all processes", "[collective][gather]")
{
    auto root = GENERATE(range(0, mpi::global::size));

    /**
     * Tests whether a single scalar value can be gathered from all processes into
     * any process as the root for the operation.
     * @since 1.0
     */
    SECTION("a single scalar value") {
        int value = mpi::global::rank + 1;
        auto result = mpi::gather(&value, 1, root, mpi::world, mpi::flag::uniform_t());

        if (root == mpi::global::rank) {
            REQUIRE(result.count == (size_t) mpi::global::size);
            for (int i = 0; i < mpi::global::size; ++i)
                REQUIRE(result[i] == (i + 1));
        } else {
            REQUIRE(result.count == 0);
        }
    }

    /**
     * Tests whether a container with an uniform amount of elements across all processes
     * can be gathered into any process as the root for the operation.
     * @since 1.0
     */
    SECTION("a uniform container of scalar values") {
        constexpr int quantity = 4;
        std::vector<int> value (quantity);

        for (int i = 0; i < quantity; ++i)
            value[i] = 10 * mpi::global::rank + i;

        auto result = mpi::gather(value, root, mpi::world, mpi::flag::uniform_t());

        if (root == mpi::global::rank) {
            REQUIRE(result.count == (size_t) (quantity * mpi::global::size));
            for (int i = 0, k = 0; i < mpi::global::size; ++i)
                for (int j = 0; j < quantity; ++j, ++k)
                    REQUIRE(result[k] == (10 * i + j));
        } else {
            REQUIRE(result.count == 0);
        }
    }

    /**
     * Tests whether a container with a varying amount of elements between processes
     * can be gathered into any process as the root for the operation.
     * @since 1.0
     */
    SECTION("a varying container of scalar values") {
        std::vector<int> value (mpi::global::rank + 1);

        for (int i = 0; i <= mpi::global::rank; ++i)
            value[i] = 100 * root + mpi::global::rank * 10 + i;

        auto result = mpi::gather(value, root, mpi::world, mpi::flag::varying_t());

        if (root == mpi::global::rank)  {
            auto size = mpi::global::size;
            REQUIRE(result.count == (size_t) (size * (size + 1) / 2));
            for (int i = 0, k = 0; i < mpi::global::size; ++i)
                for (int j = 0; j <= i; ++j, ++k)
                    REQUIRE(result[k] == (100 * mpi::global::rank + i * 10 + j));
        } else {
            REQUIRE(result.count == 0);
        }
    }
}
