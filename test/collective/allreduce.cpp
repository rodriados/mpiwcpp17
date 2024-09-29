/**
 * A thin C++17 wrapper for MPI.
 * @file Test cases for the all-reduce collective operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2023-present Rodrigo Siqueira
 */
#include <cstdint>
#include <vector>

#include <catch.hpp>
#include <mpiwcpp17.h>

SCENARIO("reduce values into all processes", "[collective][allreduce]")
{
    auto sumUpTo = [](auto n) { return (n * (n + 1)) / 2; };

    /**
     * Tests whether a single scalar value can be reduced into all processes using
     * the given functor as operator.
     * @since 1.0
     */
    GIVEN("a single scalar value") {
        int value = mpi::global::rank + 1;
        int result = mpi::allreduce(&value, 1, mpi::functor::add);

        THEN("all processes have the result") {
            int expected = sumUpTo(mpi::global::size);
            REQUIRE(result == expected);
        }
    }

    /**
     * Tests whether a container with an uniform amount of elements across all processes
     * can be reduced into all processes using the given functor as operator.
     * @since 1.0
     */
    GIVEN("a uniform container of scalar values") {
        constexpr int quantity = 4;
        std::vector<int> value (quantity);

        for (int i = 0; i < quantity; ++i)
            value[i] = (mpi::global::rank + 1) * (i + 1);

        const auto f = [](auto x, auto y) { return x + y; };
        auto result = mpi::allreduce(value, f);

        THEN("all processes have the results") {
            REQUIRE(result.count == quantity);
            for (int i = 0; i < quantity; ++i) {
                int expected = sumUpTo(mpi::global::size) * (i + 1);
                REQUIRE(result[i] == expected);
            }
        }
    }
}
