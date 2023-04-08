/**
 * A thin C++17 wrapper for MPI.
 * @file Test cases for the barrier collective operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#include <catch.hpp>
#include <mpiwcpp17.h>

/**
 * Tests whether all processes can synchronize and wait for all others to reach
 * the same point of execution within the code.
 * @since 1.0
 */
TEST_CASE("can synchronize processes", "[collective][barrier]")
{
    mpi::barrier();
    SUCCEED("processes were successfully synchronized");
}
