/**
 * A thin C++17 wrapper for MPI.
 * @file A generic type container.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <utility>
#include <mpiwcpp17/environment.h>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail
{
    /**
     * A generic container to elements of shared ownership. Besides holding a shared
     * ownership of its contents, the container also acts a transparent wrapper,
     * by allowing it to be seamlessly converted to the underlying elements.
     * @tparam T The container element type.
     * @since 2.1
     */
    template <typename T>
    struct container_t
    {
        typedef T element_t;

        T* const ptr = nullptr;
        size_t const count = 0;

        static_assert(
            std::is_trivially_copyable_v<T>
          , "only trivially copyable types can be sent via MPI");

        MPIWCPP17_CONSTEXPR container_t() noexcept = default;
        MPIWCPP17_CONSTEXPR container_t(const container_t&) noexcept = default;
        MPIWCPP17_CONSTEXPR container_t(container_t&&) noexcept = default;

        /**
         * Initializes a new container with a given count of elements.
         * @param ptr The pointer wrapped by the container.
         * @param count The total number of elements carried by the container.
         */
        MPIWCPP17_CONSTEXPR container_t(T *ptr, size_t count) noexcept
          : ptr (ptr)
          , count (count)
        {}

        MPIWCPP17_INLINE container_t& operator=(const container_t&) noexcept = delete;
        MPIWCPP17_INLINE container_t& operator=(container_t&&) noexcept = delete;

        /**#@+
         * The container's initial iterator position.
         * @return The pointer to the first element in container.
         */
        MPIWCPP17_INLINE       T *begin() noexcept       { return ptr; }
        MPIWCPP17_INLINE const T *begin() const noexcept { return ptr; }
        /**#@-*/

        /**#@+
         * The container's final iterator position.
         * @return The pointer to after the last element in container.
         */
        MPIWCPP17_INLINE       T *end() noexcept       { return ptr + count; }
        MPIWCPP17_INLINE const T *end() const noexcept { return ptr + count; }
        /**#@-*/

        /**@+
         * Seamlessly converts the container into its first element.
         * @return The container's first element.
         */
        MPIWCPP17_INLINE operator       T&() noexcept       { return first(); }
        MPIWCPP17_INLINE operator const T&() const noexcept { return first(); }
        /**#@-*/

        /**#@+
         * Converts the container into the raw contents pointer.
         * @return The pointer to container's contents.
         */
        MPIWCPP17_INLINE operator       T*() noexcept       { return ptr; }
        MPIWCPP17_INLINE operator const T*() const noexcept { return ptr; }
        /**#@-*/

        /**#@+
         * Explicitly converts the container into a boolean value. The container
         * is considered truthy if the number of elements is not zero.
         * @return Does the container hold any content?
         */
        MPIWCPP17_INLINE explicit operator bool() noexcept       { return count; }
        MPIWCPP17_INLINE explicit operator bool() const noexcept { return count; }
        /**#@-*/

        /**
         * Exposes the container's contents by an index.
         * @param i The index of the element to be accessed.
         * @return A reference to the element at the given index.
         */
        MPIWCPP17_INLINE T& operator[](ptrdiff_t i) const noexcept { return ptr[i]; }

        /**#@+
         * Exposes the container's first element.
         * @return The container's first element.
         */
        MPIWCPP17_INLINE       T& first() noexcept       { return ptr[0]; }
        MPIWCPP17_INLINE const T& first() const noexcept { return ptr[0]; }
        /**#@-*/

        /**#@+
         * Exposes the container's last element.
         * @return The container's last element.
         */
        MPIWCPP17_INLINE       T& last() noexcept       { return ptr[count - 1]; }
        MPIWCPP17_INLINE const T& last() const noexcept { return ptr[count - 1]; }
        /**#@-*/
    };
}

MPIWCPP17_END_NAMESPACE
