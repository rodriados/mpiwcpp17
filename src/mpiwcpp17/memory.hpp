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
#include <mpiwcpp17/flag.hpp>
#include <mpiwcpp17/guard.hpp>
#include <mpiwcpp17/global.hpp>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/window.hpp>
#include <mpiwcpp17/info.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace memory
{
    /**
     * Allocates speciallized memory for RMA operations. The allocated memory is
     * not shared between processes and can not be accessed directly by other processes.
     * @tparam T The type of the elements to store in the allocated memory region.
     * @param count The number of elements or bytes to allocate memory for.
     * @param info The key-value information instance to attach to allocated memory.
     * @return The smart pointer to the allocated memory.
     */
    template <typename T = void>
    MPIWCPP17_INLINE auto allocate(size_t count = 1, info_t info = info::null)
    {
        using element_t = std::conditional_t<std::is_void_v<T>, uint8_t, T>;
        using memory_t  = std::conditional_t<std::is_void_v<T>, void, element_t[]>;

        auto ptr = MPIWCPP17_GUARD_CALL(T*, MPI_Alloc_mem(count * sizeof(element_t), info, &_));
        auto d = [](T *ptr) { guard(MPI_Free_mem(ptr)); };

        return std::unique_ptr<memory_t, decltype(d)>(ptr, d);
    }

    /**
     * Allocates shared memory for direct message passing and RMA operations. The
     * allocated memory is shared between processes in the same communicator and
     * can be used for direct load and store accesses as well as for RMA operations.
     * The use of shared memory may be restricted to processes in the same node.
     * @tparam T The type of the elements to store in the allocated memory region.
     * @param count The number of elements or bytes to allocate memory for.
     * @param comm Communicator allowed for RMA operations with allocated memory.
     * @param info The key-value information instance to attach to allocated memory.
     * @return The smart pointer to the allocated memory.
     */
    template <typename T = void>
    MPIWCPP17_INLINE auto allocate_shared(size_t count = 1, communicator_t comm = world, info_t info = info::null)
    {
        using element_t = std::conditional_t<std::is_void_v<T>, uint8_t, T>;
        using memory_t  = std::conditional_t<std::is_void_v<T>, void, element_t[]>;

        auto [w, _] = detail::window::allocate<T>(count, comm, info, flag::window::shared_t());
        auto ptr = detail::window::query<T>(w, process::root);
        auto d = [=](auto) { window::free(w); };

        return std::unique_ptr<memory_t, decltype(d)>(ptr, d);
    }
}

MPIWCPP17_END_NAMESPACE
