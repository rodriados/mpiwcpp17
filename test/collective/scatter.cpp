/**
 * A thin C++17 wrapper for MPI.
 * @file Test cases for the scatter collective operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2023-present Rodrigo Siqueira
 */
#include <cstdint>
#include <vector>

#include <mpiwcpp17/api.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>

using namespace Catch;

TEST_CASE("scatter values to all processes", "[collective][scatter]")
{
    auto root = GENERATE(range(0, mpi::global::size));

    /**
     * Tests whether a container with the number of elements being a multiple of
     * processes can be scattered from the root of the operation.
     * @since 1.0
     */
    SECTION("a uniform container of scalar values") {
        constexpr int quantity = 4;
        std::vector<int> value (root == mpi::global::rank
            ? mpi::global::size * quantity: 1);

        if (root == mpi::global::rank)
            for (int i = 0; i < mpi::global::size * quantity; ++i)
                value[i] = i;

        auto result = mpi::scatter(value, root, mpi::world, mpi::flag::uniform_t());

        REQUIRE(result.count == quantity);

        for (int i = 0; i < quantity; ++i)
            REQUIRE(result[i] == mpi::global::rank * quantity + i);
    }
}
