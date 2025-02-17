/**
 * A thin C++17 wrapper for MPI.
 * @file The main file for running the defined unit tests.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#include <mpi.h>
#include <cstdint>
#include <cstddef>
#include <climits>

#include <mpiwcpp17/api.h>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

/**
 * Tests whether the world communicator's info is correctly set up.
 * @since 1.0
 */
TEST_CASE("world communicator has correct info", "[global]")
{
    int processRank, communicatorSize;

    mpi::guard(MPI_Comm_rank(MPI_COMM_WORLD, &processRank));
    mpi::guard(MPI_Comm_size(MPI_COMM_WORLD, &communicatorSize));

    REQUIRE(mpi::global::rank == processRank);
    REQUIRE(mpi::global::size == communicatorSize);
}

/**
 * Initializes MPI machinery and runs test cases.
 * @return Were all test cases successful?
 */
int main(int argc, char **argv)
{
    mpi::initiator_t m (&argc, &argv, mpi::support::thread_level_t::serialized);

    // Starting the test run session and running the tests according to the given
    // command line arguments. Each MPI process runs its own session and the results
    // are only gathered and presented together at the end.
    auto session = Catch::Session();
    int result = session.run(argc, argv);

    return result;
}
