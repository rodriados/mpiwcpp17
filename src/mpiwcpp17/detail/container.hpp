/**
 * A thin C++17 wrapper for MPI.
 * @file A generic type container.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <memory>
#include <mpiwcpp17/environment.h>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail
{
    /**
     * A generic container to elements of shared ownership. Besides holding a shared
     * ownership of its contents, the container also acts a transparent wrapper,
     * by allowing it to be seamlessly converted to the underlying elements.
     * @tparam T The container element type.
     * @since 3.0
     */
    template <typename T>
    struct container_t
    {
        typedef T element_t;

        std::shared_ptr<T[]> ptr = nullptr;
        size_t count = 0;

        MPIWCPP17_INLINE container_t() noexcept = default;
        MPIWCPP17_INLINE container_t(const container_t&) noexcept = default;
        MPIWCPP17_INLINE container_t(container_t&&) noexcept = default;

        /**
         * Initializes a new container from a shared ownership pointer.
         * @tparam U A type convertible to the container's element type.
         * @param ptr The pointer to acquire shared ownership of.
         * @param count The total number of elements carried within the container.
         */
        template <typename U>
        MPIWCPP17_INLINE container_t(const std::shared_ptr<U>& ptr, size_t count) noexcept
          : ptr (std::static_pointer_cast<T[]>(ptr))
          , count (count)
        {}

        MPIWCPP17_INLINE container_t& operator=(const container_t&) noexcept = default;
        MPIWCPP17_INLINE container_t& operator=(container_t&&) noexcept = default;

        /**#@+
         * The container's initial iterator position.
         * @return The pointer to the first element in container.
         */
        MPIWCPP17_INLINE       T *begin() noexcept       { return ptr.get(); }
        MPIWCPP17_INLINE const T *begin() const noexcept { return ptr.get(); }
        /**#@-*/

        /**#@+
         * The container's final iterator position.
         * @return The pointer to after the last element in container.
         */
        MPIWCPP17_INLINE       T *end() noexcept       { return count + ptr.get(); }
        MPIWCPP17_INLINE const T *end() const noexcept { return count + ptr.get(); }
        /**#@-*/

        /**
         * Exposes the container's contents by an index.
         * @param index The index of the element to be accessed.
         * @return A reference to the element at the given index.
         */
        MPIWCPP17_INLINE T& operator[](ptrdiff_t index) const { return ptr[index]; }

        /**@+
         * Seamlessly converts the container into its first element.
         * @return The container's first element.
         */
        MPIWCPP17_INLINE operator       T&() noexcept       { return *ptr.get(); }
        MPIWCPP17_INLINE operator const T&() const noexcept { return *ptr.get(); }
        /**#@-*/

        /**#@+
         * Converts the container into the raw contents pointer.
         * @return The pointer to container's contents.
         */
        MPIWCPP17_INLINE operator       T*() noexcept       { return ptr.get(); }
        MPIWCPP17_INLINE operator const T*() const noexcept { return ptr.get(); }
        /**#@-*/
    };
}

MPIWCPP17_END_NAMESPACE
