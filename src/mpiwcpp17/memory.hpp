/**
 * A thin C++17 wrapper for MPI.
 * @file MPI memory allocators for RMA operations.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <cstdint>
#include <utility>
#include <memory>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/guard.hpp>
#include <mpiwcpp17/global.hpp>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/window.hpp>
#include <mpiwcpp17/info.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace memory
{
    /**
     * Allocates speciallized memory for RMA operations. The allocated memory is
     * not shared and thus can not be accessed directly by other processes.
     * @tparam T The type of the elements to store in the allocated memory region.
     * @param count The number of elements or bytes to allocate memory for each process.
     * @param info The key-value information instance to attach to allocated memory.
     * @return The smart pointer to the allocated region.
     */
    template <typename T = void>
    MPIWCPP17_INLINE auto allocate(size_t count = 1, const info_t& info = info::null)
    {
        using element_t = std::conditional_t<std::is_void_v<T>, uint8_t, T>;
        using region_t  = std::conditional_t<std::is_void_v<T>, void, element_t[]>;
        constexpr size_t size = sizeof(element_t);

        auto ptr = MPIWCPP17_GUARD_CALL(T*, MPI_Alloc_mem(count * size, info, &_));
        auto fn  = [](T* ptr) -> void { guard(MPI_Free_mem(ptr)); };

        return std::unique_ptr<region_t, decltype(fn)>(ptr, fn);
    }

    /**
     * Allocates contiguous memory for processes in a shared-memory communicator.
     * The allocated memory region can be accessed directly by address from processes
     * within the same physical node. Therefore, relative addesses can be calculated
     * from variable known by each process individually.
     * @tparam T The type of the elements to store in the allocated memory region.
     * @param count The number of elements or bytes to allocate memory for each process.
     * @param comm Communicator allowed for RMA operations with allocated memory.
     * @param info The key-value information instance to attach to allocated memory.
     * @return The smart pointer to the beginning of the shared memory region.
     * @see mpi::window::allocate_shared
     */
    template <typename T = void>
    MPIWCPP17_INLINE auto allocate_shared(
        size_t count = 1
      , const communicator_t& comm = world
      , const info_t& info = info::null
    ) {
        using element_t = std::conditional_t<std::is_void_v<T>, uint8_t, T>;
        using region_t  = std::conditional_t<std::is_void_v<T>, void, element_t[]>;

        auto [base, w] = window::allocate_shared<T>(count, comm, info);
        auto [gptr, _] = window::query_shared<T>(w, process::root);
        auto fn        = [w = std::move(w)](T*) -> void {};

        return std::unique_ptr<region_t, decltype(fn)>(gptr, std::move(fn));
    }
}

MPIWCPP17_END_NAMESPACE
