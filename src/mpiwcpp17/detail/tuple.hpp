/**
 * A thin C++17 wrapper for MPI.
 * @file A general and simple tuple abstraction implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <cstdint>
#include <utility>

#include <mpiwcpp17/environment.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail
{
    /**
     * Represents a tuple leaf, which holds one of a tuple's values.
     * @tparam I The tuple's leaf's index offset.
     * @tparam T The tuple's leaf's content type.
     * @since 1.0
     */
    template <size_t I, typename T>
    struct leaf
    {
        T value;
    };

    /**
     * A tuple represents an indexable sequential list of elements of possibly
     * different types. In comparision with a plain struct containing elements
     * of similar types, the tuple must require the same amount of memory and
     * its elements cannot be accessed by field names but offset.
     * @tparam T The tuple's sequence of element types.
     * @since 1.0
     */
    template <typename ...T>
    struct tuple : public tuple<std::make_index_sequence<sizeof...(T)>, T...>
    {
        static constexpr size_t count = sizeof...(T);
    };

    /**
     * The base tuple type.
     * @tparam I The sequence indeces for the tuple elements' types.
     * @tparam T The list of tuple elements' types.
     * @since 1.0
     */
    template <size_t ...I, typename ...T>
    struct tuple<std::index_sequence<I...>, T...> : public detail::leaf<I, T>...
    {
        /**
         * Retrieves the value of a const-qualified tuple member by its index.
         * @tparam J The requested member's index.
         * @return The const-qualified member's value.
         */
        template <size_t J>
        inline constexpr auto get() const noexcept -> decltype(auto)
        {
            return get<J>(*this);
        }

        /**
         * Retrieves the requested tuple leaf and returns its value.
         * @tparam J The requested leaf index.
         * @tparam U The type of the requested leaf member.
         * @param leaf The selected const-qualified tuple leaf member.
         * @return The const-qualified leaf's value.
         */
        template <size_t J, typename U>
        inline constexpr static const U& get(const leaf<J, U>& leaf) noexcept
        {
            return leaf.value;
        }
    };
}

MPIWCPP17_END_NAMESPACE
