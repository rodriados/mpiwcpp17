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

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/guard.hpp>
#include <mpiwcpp17/info.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace memory
{
    /**
     * Allocates speciallized memory for RMA operations.
     * The allocated memory is not shared and must be manually freed by the user.
     * @tparam T The type of the elements to store in the allocated memory region.
     * @param count The number of elements or bytes to allocate memory for each process.
     * @param info The key-value information instance to attach to allocated memory.
     * @return The smart pointer to the allocated region.
     */
    template <typename T = void>
    MPIWCPP17_INLINE T* allocate(size_t count = 1, const info_t& info = info::null)
    {
        constexpr size_t size = sizeof(std::conditional_t<std::is_void_v<T>, uint8_t, T>);
        return MPIWCPP17_GUARD_CALL(T*, MPI_Alloc_mem(count * size, info, &_));
    }

    /**
     * Frees a manually allocated RMA-specialized memory region.
     * Destructors must be manually called before freeing the memory, if necessary.
     * @tparam T The type of the elements to stored the given memory region.
     * @param ptr The pointer to the memory region to be released.
     * @see mpi::memory::allocate
     */
    template <typename T = void>
    MPIWCPP17_INLINE void free(T *ptr)
    {
        return MPIWCPP17_GUARD_EVAL(MPI_Free_mem(ptr));
    }
}

MPIWCPP17_END_NAMESPACE
