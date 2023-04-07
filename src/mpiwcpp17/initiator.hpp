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
struct initiator_t
{
    const thread_support_t thread_mode;

    inline initiator_t(const initiator_t&) noexcept = delete;
    inline initiator_t(initiator_t&&) noexcept = delete;

    inline initiator_t& operator=(const initiator_t&) noexcept = delete;
    inline initiator_t& operator=(initiator_t&&) noexcept = delete;

    /**
     * Initializes the internal MPI state and processes communication.
     * @param mode The desired process thread support level.
     */
    inline initiator_t(thread_support_t mode = thread_support_t::single)
      : thread_mode (mpiwcpp17::initialize(mode))
    {}

    /**
     * Initializes the internal MPI state and processes communication with arguments.
     * @param argc The number of command line arguments to initialize MPI with.
     * @param argv The list of processes' command line arguments.
     * @param mode The desired process thread support level.
     */
    inline initiator_t(int *argc, char ***argv, thread_support_t mode = thread_support_t::single)
      : thread_mode (mpiwcpp17::initialize(argc, argv, mode))
    {}

    /**
     * Terminates the global MPI state and communication between processes.
     * @see mpi::finalize
     */
    inline ~initiator_t()
    {
        mpiwcpp17::finalize();
    }
};

MPIWCPP17_END_NAMESPACE
