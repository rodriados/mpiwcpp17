/**
 * A thin C++17 wrapper for MPI.
 * @file MPI special memory allocation.
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

MPIWCPP17_BEGIN_NAMESPACE

/*
 * In some systems, message-passing and remote-memory-access operations run faster
 * when accessing specially allocated memory, for instance memory that is shared
 * between processes in the communication group. Therefore, MPI provides an special
 * mechanism for allocating and freeing such special memory. The use of such memory
 * for performing these operations is not mandatory, and this memory can be used
 * without restrictions as any other dynamically allocated memory. However, some
 * MPI implementations may restrict the use of windows to such memory regions only.
 */

namespace memory
{
    /**
     * Allocates memory for message passing and RMA operations. The allocated memory
     * may be used as any other memory region not shared between different processes.
     * @tparam T The type of the elements to store in the allocated memory region.
     * @param The number of elements or bytes to allocate memory for.
     * @return The smart pointer to the allocated memory region.
     */
    template <typename T = void>
    MPIWCPP17_INLINE auto allocate(size_t count = 1)
    {
        T *ptr;

        using element_t = std::conditional_t<std::is_void_v<T>, uint8_t, T>;
        using memory_t  = std::conditional_t<std::is_void_v<T>, void, element_t[]>;

        // The memory allocated by MPI should only be freed using its corresponding
        // free function, as MPI might perform extra steps while releasing the memory
        // region and any related possible internal resources.
        using deleter_t = struct deleter_t {
            MPIWCPP17_INLINE void operator()(T *ptr) {
                guard(MPI_Free_mem(ptr));
            }
        };

        guard(MPI_Alloc_mem(count * sizeof(element_t), MPI_INFO_NULL, &ptr));
        return std::unique_ptr<memory_t, deleter_t>(ptr);
    }
}

MPIWCPP17_END_NAMESPACE
