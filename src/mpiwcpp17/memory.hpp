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

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/guard.hpp>

MPIWCPP17_BEGIN_NAMESPACE

/*
 * In some systems, message-passing and remote-memory-access operations run faster
 * when accessing specially allocated memory, for instance memory that is shared
 * between processes in the communication group. Therefore, MPI provides an special
 * mechanism for allocating and freeing such special memory. The use of such memory
 * for performing these operations is not mandatory, and this memory can be used
 * without restrictions as any other dynamically allocated memory. However, some
 * MPI implementations may restrict the use of windows within such memory only.
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
    inline auto allocate(size_t count = 1) -> decltype(auto)
    {
        T *ptr = nullptr;
        auto destructor = [](T *wptr) { guard(MPI_Free_mem(wptr)); };

        using destructor_t = void(*)(T*);

        if constexpr (!std::is_void<T>::value) {
            guard(MPI_Alloc_mem(count * sizeof(T), MPI_INFO_NULL, &ptr));
            return std::unique_ptr<T[], destructor_t>(ptr, destructor);

        } else {
            guard(MPI_Alloc_mem(count, MPI_INFO_NULL, &ptr));
            return std::unique_ptr<void, destructor_t>(ptr, destructor);
        }
    }
}

MPIWCPP17_END_NAMESPACE
