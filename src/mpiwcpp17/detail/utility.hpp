/**
 * A thin C++17 wrapper for MPI.
 * @file The MPI operations utility functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2025-present Rodrigo Siqueira
 */
#pragma once

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/detail/payload.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail
{
    /**
     * Auxiliary iterable argument receptor.
     * This type sole purpose is to enable generic iterable types to be seamlessly
     * used as input payload types when calling an operation function.
     * @tparam T The type of argument to receive.
     * @since 2.1
     */
    template <typename T>
    struct iterable_t : public payload_const_t<T>
    {
        using payload_const_t<T>::payload_const_t;

        /**
         * Creates an input payload from a generic container.
         * @tparam C The generic container type.
         * @param container The generic container instance.
         */
        template <typename C>
        MPIWCPP17_INLINE iterable_t(C&& container)
          : payload_const_t<T> (payload::to_tentative_input(std::forward<C>(container)))
        {}

        /**
         * Creates an input payload from an initializer list.
         * @tparam U The initializer list contents type.
         * @param list The initializer list to create a payload from.
         */
        template <typename U>
        MPIWCPP17_INLINE iterable_t(const std::initializer_list<U>& list)
          : payload_const_t<T> (payload::to_tentative_input(list))
        {}
    };
}

MPIWCPP17_END_NAMESPACE
