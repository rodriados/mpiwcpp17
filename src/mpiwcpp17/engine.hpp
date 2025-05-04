/**
 * A thin C++17 wrapper for MPI.
 * @file The RAII initiator for global MPI machinery.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/exception.hpp>
#include <mpiwcpp17/support.hpp>

#include <mpiwcpp17/detail/world.hpp>
#include <mpiwcpp17/detail/raii.hpp>

MPIWCPP17_BEGIN_NAMESPACE

/**
 * Automatically initializes and finalizes the global MPI engine and state. This
 * type should be used as a global singleton and should only be instatiated once.
 * @since 1.0
 */
struct engine_t final
{
    MPIWCPP17_INLINE engine_t(const engine_t&) noexcept = delete;
    MPIWCPP17_INLINE engine_t(engine_t&&) noexcept = delete;

    /**
     * Initializes the internal MPI engine and processes communication with arguments.
     * @param argc The number of arguments to initialize spawning processes with.
     * @param argv The list of arguments to initialize spawning processes with.
     * @param mode The desired process thread-support level.
     */
    MPIWCPP17_INLINE engine_t(
        int *argc, char ***argv
      , support::thread_level_t mode = support::thread_level_t::single
    ) {
        if (detail::world_t::initialized() || detail::world_t::finalized())
            throw exception_t("MPI cannot be initialized");
        detail::world_t::initialize(argc, argv, mode);
    }

    /**
     * Initializes the internal MPI engine and processes communication.
     * @param mode The desired process thread-support level.
     */
    MPIWCPP17_INLINE engine_t(support::thread_level_t mode = support::thread_level_t::single)
      : engine_t (nullptr, nullptr, mode)
    {}

    MPIWCPP17_INLINE engine_t& operator=(const engine_t&) noexcept = delete;
    MPIWCPP17_INLINE engine_t& operator=(engine_t&&) noexcept = delete;

    /**
     * Finalizes MPI, cleans-up resources and closes all processes communications.
     * @see mpi::engine_t::engine_t
     */
    MPIWCPP17_INLINE ~engine_t()
    {
        if (!detail::world_t::finalized()) {
            detail::raii_t::finalize();
            detail::world_t::finalize();
        }
    }
};

MPIWCPP17_END_NAMESPACE
