/**
 * A thin C++17 wrapper for MPI.
 * @file A pointer wrapper for collective operations.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <algorithm>
#include <iterator>
#include <utility>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/datatype.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail
{
    /**
     * Wraps a non-owning pointer for collective operations.
     * @tparam T The wrapped pointer's type.
     * @since 2.1
     */
    template <typename T>
    struct wrapper_t
    {
        typedef T element_t;
        typedef element_t *pointer_t;

        pointer_t ptr = nullptr;
        datatype_t type = datatype::identify<element_t>();
        size_t count = 0;

        static_assert(
            std::is_trivially_copyable<T>::value
          , "only trivially copyable types can be used with MPI");

        inline wrapper_t() noexcept = delete;
        inline wrapper_t(wrapper_t&) noexcept = default;
        inline wrapper_t(wrapper_t&&) noexcept = default;

        /**
         * Wraps a simple value into a pointer wrapper.
         * @param value The value to be wrapped.
         */
        inline wrapper_t(element_t& value) noexcept
          : wrapper_t (&value, 1)
        {}

        /**
         * Wraps a raw non-owning pointer.
         * @param ptr The pointer to be wrapped.
         * @param count The total number of elements carried by the pointer.
         */
        inline wrapper_t(pointer_t ptr, size_t count = 1) noexcept
          : ptr (ptr)
          , count (count)
        {}

        /**
         * Wraps an instance of a contiguous memory container.
         * @tparam C The type of container to be wrapped.
         * @param container The container instance to be wrapped.
         * @param count The number of elements to wrap from container.
         */
        template <
            template <class> class C
          , typename = typename std::enable_if<std::is_same<
                typename std::iterator_traits<
                    decltype(std::begin(std::declval<C<element_t>&>()))
                >::iterator_category
              , std::random_access_iterator_tag
            >::value>::type
        >
        inline wrapper_t(C<element_t>& container, size_t count = 0)
          : wrapper_t (
                &*std::begin(container)
              , count ? count : std::distance(std::begin(container), std::end(container))
            )
        {}

        inline wrapper_t& operator=(const wrapper_t&) = delete;
        inline wrapper_t& operator=(wrapper_t&&) = delete;

        /**
         * Unwraps the const-qualified pointer.
         * @return The wrapped pointer.
         */
        inline operator const pointer_t() const noexcept
        {
            return ptr;
        }
    };

    /*
     * Deduction guides for pointer wrappers.
     * @since 2.1
     */
    template <template <class> class C, typename T> wrapper_t(C<T>&, size_t = 0) -> wrapper_t<T>;
}

MPIWCPP17_END_NAMESPACE
