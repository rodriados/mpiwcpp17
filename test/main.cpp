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

#define CATCH_CONFIG_RUNNER
#define CATCH_CONFIG_DEFAULT_REPORTER "mpi-reporter"

#include <catch.hpp>
#include <mpiwcpp17.h>

static int processRank = -1;
static int communicatorSize = -1;

static MPI_Datatype totals_datatype;
static MPI_Op totals_sum_op;

/**
 * Tests whether the world communicator's info is correctly set up.
 * @since 1.0
 */
TEST_CASE("world communicator has correct info", "[global]")
{
    REQUIRE(mpi::world.rank == processRank);
    REQUIRE(mpi::world.size == communicatorSize);
}

/**
 * Specialization of console reporter to use with an MPI application. This reporter
 * will reduce the final test run statistics to the root process, in order to show
 * a sum of runs in all processes at the same time.
 * @since 1.0
 */
struct MPIConsoleReporter : Catch::ConsoleReporter
{
    using Catch::ConsoleReporter::ConsoleReporter;

    /**
     * Reports to the console the statistics of the current test run in all processes.
     * @param stats The test run statistics of the current process.
     */
    void testRunEnded(const Catch::TestRunStats& stats) override
    {
        Catch::Totals reducedTotals;
        MPI_Reduce(&stats.totals, &reducedTotals, 1, totals_datatype, totals_sum_op, 0, MPI_COMM_WORLD);

        if (processRank != 0)
            return;

        auto reducedStats = Catch::TestRunStats(stats.runInfo, reducedTotals, stats.aborting);
        Catch::ConsoleReporter::testRunEnded(reducedStats);
    }

    /**
     * Reports to the console that an assertion has failed, or a successful assertion
     * if so required by given test run configuration .
     * @param stats The assertion statistics to be reported.
     * @return Should the console buffer be cleared?
     */
    bool assertionEnded(const Catch::AssertionStats& stats) override
    {
        const auto& result = stats.assertionResult;

        bool shouldClearBuffer = false;
        bool includeResults = !result.isOk() || this->m_config->includeSuccessfulResults();

        for (int i = 0; i < communicatorSize; ++i) {
            if (processRank == i && (includeResults || result.getResultType() == Catch::ResultWas::Warning)) {
                auto printer = Catch::ConsoleAssertionPrinter(this->stream, stats, includeResults);
                auto colour = Catch::Colour(Catch::Colour::FileName);

                this->stream << "[process #" << processRank << "] ";
                printer.print();

                this->stream << std::endl << std::flush;
                shouldClearBuffer = true;
            }
        }

        return shouldClearBuffer;
    }
};

CATCH_REGISTER_REPORTER("mpi-reporter", MPIConsoleReporter);

/**
 * Gets the correct MPI datatype value for a size_t value.
 * @return The MPI datatype for a size_t value.
 */
inline static constexpr MPI_Datatype MPI_sizeT()
{
    if constexpr (SIZE_MAX == UCHAR_MAX) {
        return MPI_UNSIGNED_CHAR;
    } else if constexpr (SIZE_MAX == USHRT_MAX) {
        return MPI_UNSIGNED_SHORT;
    } else if constexpr (SIZE_MAX == UINT_MAX) {
        return MPI_UNSIGNED;
    } else if constexpr (SIZE_MAX == ULONG_MAX) {
        return MPI_UNSIGNED_LONG;
    } else {
        return MPI_UNSIGNED_LONG_LONG;
    }
}

/**
 * Initializes MPI machinery and runs test cases.
 * @return Were all test cases successful?
 */
int main(int argc, char **argv)
{
    int flag, provided;
    MPI_Datatype counts_datatype;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    MPI_Comm_size(MPI_COMM_WORLD, &communicatorSize);

    // Creating the Catch::Counts type structure for MPI. This is needed to allow
    // this type to be sent via MPI messages.
    int counts_blocks[3] = {1, 1, 1};
    MPI_Datatype counts_types[3] = {MPI_sizeT(), MPI_sizeT(), MPI_sizeT()};
    MPI_Aint counts_displs[3] = {
        offsetof(Catch::Counts, passed)
      , offsetof(Catch::Counts, failed)
      , offsetof(Catch::Counts, failedButOk)
    };

    MPI_Type_create_struct(3, counts_blocks, counts_displs, counts_types, &counts_datatype);
    MPI_Type_commit(&counts_datatype);

    // Creating the Catch::Totals type structure for MPI. This is needed to allow
    // this type to be sent via MPI messages.
    int totals_blocks[3] = {1, 1, 1};
    MPI_Datatype totals_types[3] = {MPI_INT, counts_datatype, counts_datatype};
    MPI_Aint totals_displs[3] = {
        offsetof(Catch::Totals, error)
      , offsetof(Catch::Totals, assertions)
      , offsetof(Catch::Totals, testCases)
    };

    MPI_Type_create_struct(3, totals_blocks, totals_displs, totals_types, &totals_datatype);
    MPI_Type_commit(&totals_datatype);

    // Creating the MPI operator for summing a group of Catch::Totals instances.
    // This operator can be used in a reduce operation to sum the values of all
    // processes in one go.
    MPI_Op_create([](void *a, void *b, int *length, MPI_Datatype*) {
        auto x = reinterpret_cast<Catch::Totals *>(a);
        auto y = reinterpret_cast<Catch::Totals *>(b);

        for (int i = 0; i < *length; ++i)
            y[i] += x[i];

    }, true, &totals_sum_op);

    mpi::init();

    // Starting the test run session and running the tests according to the given
    // command line arguments. Each MPI process runs its own session and the results
    // are only gathered and presented together at the end.
    auto session = Catch::Session();
    int result = session.run(argc, argv);

    MPI_Type_free(&totals_datatype);
    MPI_Type_free(&counts_datatype);
    MPI_Op_free(&totals_sum_op);

    mpi::finalize();

    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) MPI_Finalize();

    return result;
}
