/**
 * A thin C++17 wrapper for MPI.
 * @file MPI shared memory and RMA functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2025-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <memory>
#include <cstdint>
#include <utility>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/guard.hpp>
#include <mpiwcpp17/global.hpp>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/datatype.hpp>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/info.hpp>

#include <mpiwcpp17/detail/attribute.hpp>
#include <mpiwcpp17/detail/handle.hpp>

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

/**
 * The raw MPI window reference type.
 * This type is used to identify memory windows managed by MPI that are available
 * for performing RMA operations.
 * @since 2.1
 */
struct window_t : MPIWCPP17_INHERIT_HANDLE(MPI_Win, MPI_Win_free);

/*
 * Auxiliary macros for implementing functions that wrap the creation of new windows.
 * The newly created windows are automatically attached to RAII.
 * @param x The window to be attached to RAII.
 * @param B The call block to be wrapped.
 */
#define MPIWCPP17_WIN_RAII(x)  window_t((x), true)
#define MPIWCPP17_WIN_CALL(B)  MPIWCPP17_WIN_RAII(MPIWCPP17_GUARD_CALL(MPI_Win, B))

namespace window
{
    /**
     * The invalid or empty window instance.
     * This can be used to verify whether a window is not in a valid state or to
     * denote an empty or uninitialized window.
     * @since 2.1
     */
    MPIWCPP17_INLINE const window_t null = MPI_WIN_NULL;

    /**
     * Declares window attribute namespace and corresponding functions.
     * Attributes are identified by keys that can be used to attach and retrieve
     * generic data from window instances.
     * @since 2.1
     */
    MPIWCPP17_ATTRIBUTE_DECLARE(
        window_t
      , MPI_Win_create_keyval, MPI_Win_free_keyval
      , MPI_Win_get_attr, MPI_Win_set_attr, MPI_Win_delete_attr
      , MPI_WIN_DUP_FN, MPI_WIN_NULL_DELETE_FN
    )

    /**
     * The window synchronization modes. These modes can be used to specify the type
     * of synchronization to be performed on a window when calling the corresponding
     * synchronization functions.
     * @since 2.1
     */
    enum mode_t : int
    {
        /**
         * No special synchronization mode. This is the default synchronization mode
         * for all synchronization functions, no guarantees given whatsoever.
         * @since 2.1
         */
        none       = 0

        /**
         * The mode for no preceding RMA operations. This mode indicates that no
         * RMA operations will be issued before synchronization.
         * @since 2.1
         */
      , no_precede = MPI_MODE_NOPRECEDE

        /**
         * The mode for no preceding local stores or get calls. This mode indicates
         * that no local stores or get calls will be issued before synchronization.
         * @since 2.1
         */
      , no_store   = MPI_MODE_NOSTORE

        /**
         * The mode for no updates by any put or accumulate calls. This mode indicates
         * that no put or accumulate calls will be performed until the next synchronization.
         * @since 2.1
         */
      , no_put     = MPI_MODE_NOPUT

        /**
         * The mode for no succeeding RMA operations. This mode indicates that no
         * RMA operations will be issued after synchronization.
         * @since 2.1
         */
      , no_succeed = MPI_MODE_NOSUCCEED
    };

    /**
     * The window lock types. These types can be used to specify the type of lock
     * to be acquired on a window when calling the corresponding lock functions.
     * @since 2.1
     */
    enum lock_t : int
    {
        /**
         * The exclusive lock type. This lock type indicates that the calling process
         * will acquire an exclusive lock on the window, which prevents any other
         * process from performing access operations on the window.
         * @since 2.1
         */
        exclusive = MPI_LOCK_EXCLUSIVE

        /**
         * The shared lock type. This lock type indicates that the calling process
         * will acquire a shared lock on the window, which allows other processes
         * to perform access operations as long as they also acquire a shared lock.
         * @since 2.1
         */
      , shared    = MPI_LOCK_SHARED
    };

    /**
     * Allocates memory for each process in the communicator, managed by a new window
     * instance enabling RMA communication. The amount of memory allocated may differ
     * between processes and is determined individually by its own parameters.
     * @tparam T The type of the elements to store in the allocated memory region.
     * @param count The number of elements or bytes to allocate for each process.
     * @param comm Communicator allowed for RMA operations with allocated memory.
     * @param info The key-value information instance to attach to allocated memory.
     * @return The allocated memory pointer and new window instance.
     */
    template <typename T = void>
    MPIWCPP17_INLINE std::pair<T*, window_t> allocate(
        size_t count = 1
      , const communicator_t& comm = world
      , const info_t& info = info::null
    ) {
        constexpr size_t size = sizeof(std::conditional_t<std::is_void_v<T>, uint8_t, T>);
        auto result = MPIWCPP17_GUARD_CALL(
            struct { T *ptr; window_t::raw_t w; }
          , MPI_Win_allocate(count * size, size, info, comm, &_.ptr, &_.w));
        return std::make_pair(result.ptr, MPIWCPP17_WIN_RAII(result.w));
    }

    /**
     * Allocates shared memory for the processes of a shared-memory communicator.
     * The amount of memory allocated by each process may differ but the allocated
     * regions are guaranteed to be contiguous to the neighboring processes, according
     * to each process' rank in the communicator.
     * @tparam T The type of the elements to store in the allocated memory region.
     * @param count The number of elements or bytes to allocate for each process.
     * @param comm Communicator allowed for RMA operations with allocated memory.
     * @param info The key-value information instance to attach to allocated memory.
     * @return The allocated memory pointer and new window instance.
     */
    template <typename T = void>
    MPIWCPP17_INLINE std::pair<T*, window_t> allocate_shared(
        size_t count = 1
      , const communicator_t& comm = world
      , const info_t& info = info::null
    ) {
        constexpr size_t size = sizeof(std::conditional_t<std::is_void_v<T>, uint8_t, T>);
        auto result = MPIWCPP17_GUARD_CALL(
            struct { T *ptr; window_t::raw_t w; }
          , MPI_Win_allocate_shared(count * size, size, info, comm, &_.ptr, &_.w));
        return std::make_pair(result.ptr, MPIWCPP17_WIN_RAII(result.w));
    }

    /**
     * Queries the shared memory region associated with a process in the window.
     * @tparam T The type of the elements stored in the allocated memory region.
     * @param win The window instance of the shared memory region.
     * @param rank The process rank to query the shared memory region of.
     * @return The process-local address and size of the queried memory region.
     */
    template <typename T = void>
    MPIWCPP17_INLINE std::pair<T*, size_t> query_shared(
        const window_t& win
      , process_t rank = process::root
    ) {
        auto query = MPIWCPP17_GUARD_CALL(
            struct { T *ptr; MPI_Aint size; int displ; }
          , MPI_Win_shared_query(win, rank, &_.size, &_.displ, &_.ptr));
        return std::make_pair(query.ptr, static_cast<size_t>(query.size));
    }

    /**
     * Creates a new window to manage the specified memory region.
     * @tparam T The type of the elements stored in the memory region.
     * @param ptr The pointer to the memory region to be managed by the new window.
     * @param count The number of elements or bytes in the memory region.
     * @param comm Communicator over which the RMA operations will be executed.
     * @param info The key-value information instance to attach to the new window.
     * @return The new window instance that manages the specified memory region.
     */
    template <typename T = void>
    MPIWCPP17_INLINE window_t create(
        T *ptr, size_t count = 1
      , const communicator_t& comm = world
      , const info_t& info = info::null
    ) {
        constexpr size_t size = sizeof(std::conditional_t<std::is_void_v<T>, uint8_t, T>);
        return MPIWCPP17_WIN_CALL(MPI_Win_create(ptr, count * size, size, info, comm, &_));
    }

    /**
     * Creates a new window to manage a dynamically attached memory region.
     * @param comm Communicator over which the RMA operations will be executed.
     * @param info The key-value information instance to attach to the new window.
     * @return The new window instance that manages a memory regions attached dynamically.
     */
    MPIWCPP17_INLINE window_t create(
        const communicator_t& comm = world
      , const info_t& info = info::null
    ) {
        return MPIWCPP17_WIN_CALL(MPI_Win_create_dynamic(info, comm, &_));
    }

    /**
     * Attaches a memory region to a window.
     * @tparam T The type of the elements stored in the memory region to be attached.
     * @param win The window instance to attach the memory region to.
     * @param ptr The pointer to the memory region to be attached to the window.
     * @param count The number of elements or bytes in the memory region to be attached.
     */
    template <typename T = void>
    MPIWCPP17_INLINE void attach(const window_t& win, T *ptr, size_t count = 1)
    {
        constexpr size_t size = sizeof(std::conditional_t<std::is_void_v<T>, uint8_t, T>);
        guard(MPI_Win_attach(win, ptr, count * size));
    }

    /**
     * Detaches a memory region from a window.
     * @tparam T The type of the elements stored in the memory region to be detached.
     * @param win The window instance to detach the memory region from.
     * @param ptr The pointer to the memory region to be detached from the window.
     */
    template <typename T = void>
    MPIWCPP17_INLINE void detach(const window_t& win, T *ptr)
    {
        guard(MPI_Win_detach(win, ptr));
    }

    /**
     * Synchronizes RMA operations on a window by blocking the calling process until
     * all RMA operations issued by it on the window are completed and the window
     * is ready for the next RMA access epoch.
     * @param win The window instance to synchronize RMA operations on.
     * @param mode The window's synchronization mode.
     * @see mpi::window::mode_t
     */
    MPIWCPP17_INLINE void fence(const window_t& win, mode_t mode = mode_t::none)
    {
        guard(MPI_Win_fence((int) mode, win));
    }

    /**
     * Synchronizes RMA operations on a window by blocking the process until all
     * RMA operations issued on the window are completed.
     * @param win The window instance to synchronize RMA operations on.
     */
    MPIWCPP17_INLINE void flush(const window_t& win)
    {
        guard(MPI_Win_flush_all(win));
    }

    /**
     * Synchronizes RMA operations on a window by blocking the process until all
     * RMA operations issued by the given process on the window are completed.
     * @param win The window instance to synchronize RMA operations on.
     * @param rank The rank of the process to synchronize RMA operations with.
     */
    MPIWCPP17_INLINE void flush(const window_t& win, process_t rank)
    {
        guard(MPI_Win_flush(rank, win));
    }

    /**
     * Starts an RMA access epoch locking access to a particular process.
     * @param win The window instance to acquire the lock on.
     * @param rank The rank of the process to acquire the lock with.
     * @param lock The type of lock to be acquired.
     * @param mode The window's synchronization mode.
     */
    MPIWCPP17_INLINE void lock(
        const window_t& win
      , process_t rank
      , lock_t lock
      , mode_t mode = mode_t::none
    ) {
        guard(MPI_Win_lock(lock, rank, (int) mode, win));
    }

    /**
     * Starts an RMA access epoch locking access to all processes in window.
     * @param win The window instance to acquire the lock on.
     * @param mode The window's synchronization mode.
     */
    MPIWCPP17_INLINE void lock(const window_t& win, mode_t mode = mode_t::none)
    {
        guard(MPI_Win_lock_all((int) mode, win));
    }

    /**
     * Completes an RMA access epoch by unlocking access to a particular process.
     * @param win The window instance to release the lock on.
     * @param rank The rank of the process to release the lock with.
     */
    MPIWCPP17_INLINE void unlock(const window_t& win, process_t rank)
    {
        guard(MPI_Win_unlock(rank, win));
    }

    /**
     * Completes an RMA access epoch by unlocking access to all processes in window.
     * @param win The window instance to release the lock on.
     */
    MPIWCPP17_INLINE void unlock(const window_t& win)
    {
        guard(MPI_Win_unlock_all(win));
    }

    /**
     * Synchronizes the private and public copies of the window.
     * @param win The window instance to be synchronized.
     */
    MPIWCPP17_INLINE void sync(const window_t& win)
    {
        guard(MPI_Win_sync(win));
    }

    /**
     * Checks whether a window is empty or not. An empty window does not manage any
     * memory region and cannot be used for performing RMA operations.
     * @param win The window instance to check for emptiness.
     * @return Is the given window empty?
     */
    MPIWCPP17_INLINE bool empty(const window_t& win)
    {
        return win == window::null;
    }
}

#undef MPIWCPP17_WIN_CALL
#undef MPIWCPP17_WIN_RAII

MPIWCPP17_END_NAMESPACE
