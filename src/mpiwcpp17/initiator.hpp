/**
 * A thin C++17 wrapper for MPI.
 * @file The RAII initiator for global MPI machinery.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/support.hpp>
#include <mpiwcpp17/global.hpp>

MPIWCPP17_BEGIN_NAMESPACE

/**
 * Automatically initializes and finalizes the global MPI state. This type cannot
 * be instatiated more than once.
 * @since 1.0
 */
struct initiator_t final
{
    const support::thread_level_t thread_level;

    MPIWCPP17_INLINE initiator_t(const initiator_t&) noexcept = delete;
    MPIWCPP17_INLINE initiator_t(initiator_t&&) noexcept = delete;

    /**
     * Initializes the internal MPI machinery and processes communication with arguments.
     * @param argc The number of arguments to initialize spawning processes with.
     * @param argv The list of arguments to initialize spawning processes with.
     * @param mode The desired process thread-support level.
     */
    MPIWCPP17_INLINE initiator_t(
        int *argc, char ***argv
      , support::thread_level_t mode = support::thread_level_t::single
    ) : thread_level (initialize(argc, argv, mode))
    {}

    /**
     * Initializes the internal MPI machinery and processes communication.
     * @param mode The desired process thread-support level.
     */
    MPIWCPP17_INLINE initiator_t(support::thread_level_t mode = support::thread_level_t::single)
      : thread_level (initialize(mode))
    {}

    MPIWCPP17_INLINE initiator_t& operator=(const initiator_t&) noexcept = delete;
    MPIWCPP17_INLINE initiator_t& operator=(initiator_t&&) noexcept = delete;

    /**
     * Finalizes MPI, cleans-up resources and closes all processes communications.
     * @see mpi::finalize
     */
    MPIWCPP17_INLINE ~initiator_t()
    {
        finalize();
    }
};

MPIWCPP17_END_NAMESPACE
