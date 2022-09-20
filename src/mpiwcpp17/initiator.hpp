/**
 * A thin C++17 wrapper for MPI.
 * @file The RAII initiator for global MPI machinery.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/world.hpp>

MPIWCPP17_BEGIN_NAMESPACE

/**
 * Automatically initializes and finalizes the global MPI state. This type cannot
 * be instatiated more than once.
 * @since 1.0
 */
struct initiator
{
    const thread_support thread_mode;

    inline initiator(const initiator&) noexcept = delete;
    inline initiator(initiator&&) noexcept = delete;

    inline initiator& operator=(const initiator&) noexcept = delete;
    inline initiator& operator=(initiator&&) noexcept = delete;

    /**
     * Initializes the internal MPI state and processes communication.
     * @param mode The desired process thread support level.
     */
    inline initiator(thread_support mode = thread_support::single)
      : thread_mode (mpiwcpp17::initialize(mode))
    {}

    /**
     * Initializes the internal MPI state and processes communication with arguments.
     * @param argc The number of command line arguments to initialize MPI with.
     * @param argv The list of processes' command line arguments.
     * @param mode The desired process thread support level.
     */
    inline initiator(int *argc, char ***argv, thread_support mode = thread_support::single)
      : thread_mode (mpiwcpp17::initialize(argc, argv, mode))
    {}

    /**
     * Terminates the global MPI state and communication between processes.
     * @see mpiwcpp17::finalize
     */
    inline ~initiator()
    {
        mpiwcpp17::finalize();
    }
};

MPIWCPP17_END_NAMESPACE
