/**
 * A thin C++17 wrapper for MPI.
 * @file Test cases for the barrier collective operation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#include <catch.hpp>
#include <mpiwcpp17.h>

TEST_CASE("can synchronize processes", "[global]")
{
    mpi::barrier();
    SUCCEED("processes were successfully synchronized");
}
