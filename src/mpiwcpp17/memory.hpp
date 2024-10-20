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
     * may be used as any other memory region.
     * @tparam T The type of elements to store in the allocated memory region.
     * @param The number of elements or bytes to allocate memory for.
     * @return The smart pointer to the allocated memory region.
     */
    template <typename T = void>
    MPIWCPP17_INLINE auto allocate(size_t count = 1) -> decltype(auto)
    {
        // The memory allocated by MPI should only be freed using its corresponding
        // function, as MPI might perform extra steps while releasing the region.
        using deleter_t = struct deleter_t {
            MPIWCPP17_INLINE void operator()(T *ptr) {
                guard(MPI_Free_mem(ptr));
            }
        };

        if constexpr (!std::is_void<T>::value) {
            T *ptr; guard(MPI_Alloc_mem(count * sizeof(T), MPI_INFO_NULL, &ptr));
            return std::unique_ptr<T[], deleter_t>(ptr);

        } else {
            T *ptr; guard(MPI_Alloc_mem(count, MPI_INFO_NULL, &ptr));
            return std::unique_ptr<void, deleter_t>(ptr);
        }
    }
}

MPIWCPP17_END_NAMESPACE
