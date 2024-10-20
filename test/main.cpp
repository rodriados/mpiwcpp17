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
        Catch::Totals totals = stats.totals;
        auto reducedTotals = mpi::reduce(totals, AccumulateTotals());

        if (mpi::global::rank != 0)
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

        for (int i = 0; i < mpi::global::size; ++i) {
            if (mpi::global::rank == i && (includeResults || result.getResultType() == Catch::ResultWas::Warning)) {
                auto printer = Catch::ConsoleAssertionPrinter(this->stream, stats, includeResults);
                auto colour = Catch::Colour(Catch::Colour::FileName);

                this->stream << "[process #" << mpi::global::rank << "] ";
                printer.print();

                this->stream << std::endl << std::flush;
                shouldClearBuffer = true;
            }
        }

        return shouldClearBuffer;
    }

    /**
     * The MPI operator for accumulating the results of a test case or test run.
     * @since 1.0
     */
    struct AccumulateTotals
    {
        /**
         * Accumulates two test result count instances.
         * @param a The first result instance to be accumulated.
         * @param b The second result instance to be accumulated.
         * @return The resulting accumulated instance.
         */
        inline static Catch::Counts addCounts(const Catch::Counts& a, const Catch::Counts& b)
        {
            return Catch::Counts {
                a.passed + b.passed
              , a.failed + b.failed
              , a.failedButOk + b.failedButOk
            };
        }

        /**
         * The accumulate operator implementation.
         * @param a The first instance to accumulated.
         * @param b The second instance to accumulated.
         * @return The resulting accumulated instance.
         */
        inline Catch::Totals operator()(const Catch::Totals& a, const Catch::Totals& b)
        {
            return Catch::Totals {
                a.error + b.error
              , addCounts(a.assertions, b.assertions)
              , addCounts(a.testCases, b.testCases)
            };
        }
    };
};

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

CATCH_REGISTER_REPORTER("mpi-reporter", MPIConsoleReporter)

/**
 * Provides a datatype descriptor identifier for  the framework's test state counter.
 * @since 3.0
 */
template <>
struct mpi::datatype::provider_t<Catch::Counts> {
    inline static mpi::datatype_t provide() {
        return mpi::datatype::provide(
            &Catch::Counts::passed
          , &Catch::Counts::failed
          , &Catch::Counts::failedButOk);
    }
};

/**
 * Provides a datatype descriptor identifier for the framework's test state totalization.
 * @since 3.0
 */
template <>
struct mpi::datatype::provider_t<Catch::Totals> {
    inline static mpi::datatype_t provide() {
        return mpi::datatype::provide(
            &Catch::Totals::error
          , &Catch::Totals::assertions
          , &Catch::Totals::testCases);
    }
};

/**
 * Initializes MPI machinery and runs test cases.
 * @return Were all test cases successful?
 */
int main(int argc, char **argv)
{
    mpi::initiator_t m (&argc, &argv, mpi::support::thread_t::serialized);

    // Starting the test run session and running the tests according to the given
    // command line arguments. Each MPI process runs its own session and the results
    // are only gathered and presented together at the end.
    auto session = Catch::Session();
    int result = session.run(argc, argv);

    return result;
}
