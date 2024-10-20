/**
 * A thin C++17 wrapper for MPI.
 * @file MPI type-independent message payload for collective operations.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <utility>
#include <iterator>
#include <algorithm>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/detail/container.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail
{
    /**
     * A message payload of input data to send through a MPI collective operation.
     * In practice, this wraps an outgoing message in a neutral context for transmission.
     * @tparam The payload's message type.
     * @since 3.0
     */
    template <typename T>
    struct payload_in_t
    {
        typedef T element_t;

        const T *ptr = nullptr;
        size_t count = 0;

        static_assert(
            std::is_trivially_copyable_v<T>
          , "only trivially copyable types can be sent via MPI");

        MPIWCPP17_INLINE payload_in_t() noexcept = default;
        MPIWCPP17_INLINE payload_in_t(const payload_in_t&) noexcept = default;
        MPIWCPP17_INLINE payload_in_t(payload_in_t&&) noexcept = default;

        /**
         * Initializes a new input payload from a raw pointer.
         * @param ptr The pointer to wrap as input.
         * @param count The total number of elements carried as input.
         */
        MPIWCPP17_CONSTEXPR payload_in_t(const T *ptr, size_t count) noexcept
          : ptr (ptr)
          , count (count)
        {}

        MPIWCPP17_INLINE payload_in_t& operator=(const payload_in_t&) = delete;
        MPIWCPP17_INLINE payload_in_t& operator=(payload_in_t&&) = delete;
    };

    /**
     * A message payload of output data received through a MPI collective operation.
     * In practice, this serves as an ownership data container for incoming messages.
     * @tparam T The payload's message type.
     * @since 3.0
     */
    template <typename T>
    struct payload_out_t : public container_t<T>
    {
        using typename container_t<T>::element_t;

        static_assert(
            std::is_trivially_copyable_v<T>
          , "only trivially copyable types can be sent via MPI");

        MPIWCPP17_INLINE payload_out_t() noexcept = default;
        MPIWCPP17_INLINE payload_out_t(const payload_out_t&) noexcept = default;
        MPIWCPP17_INLINE payload_out_t(payload_out_t&&) noexcept = default;

        using container_t<T>::container_t;

        MPIWCPP17_INLINE payload_out_t& operator=(const payload_out_t&) noexcept = default;
        MPIWCPP17_INLINE payload_out_t& operator=(payload_out_t&&) noexcept = default;
    };
}

namespace detail::payload
{
    /**#@+
     * Determines whether the given type is a contiguous iterable, therefore being
     * able to be converted into an input payload.
     * @tparam T The type to test if it is a contiguous iterable.
     * @since 3.0
     */
    template <typename T, typename = void>
    MPIWCPP17_CONSTEXPR bool is_contiguous_iterable_v = false;

    template <typename T>
    MPIWCPP17_CONSTEXPR bool is_contiguous_iterable_v<T, std::enable_if_t<
        std::is_same_v<
            typename std::iterator_traits<decltype(std::begin(std::declval<T&>()))>
                ::iterator_category
          , std::random_access_iterator_tag>>> = true;
    /**#@-*/

    /**
     * Creates a new input payload from a container of contiguous elements.
     * @tparam C The type of contiguous memory container to be wrapped.
     * @param container The container instance to be wrapped.
     * @return The new input payload instance initialized from the container.
     */
    template <typename C, typename = std::enable_if_t<is_contiguous_iterable_v<C>>>
    MPIWCPP17_INLINE auto to_input(const C& container)
    {
        return payload_in_t(
            &(*std::begin(container))
          , std::distance(std::begin(container), std::end(container)));
    }

    /**
     * Creates a new input from a plain data variable.
     * @tparam T The variable type for the payload.
     * @param data The variable to be carried by the payload.
     * @return The new input payload instance.
     */
    template <typename T, typename = std::enable_if_t<!is_contiguous_iterable_v<T>>>
    MPIWCPP17_INLINE payload_in_t<T> to_input(const T& data) noexcept
    {
        return payload_in_t(&data, 1);
    }

    /**
     * Creates a new input from an output payload.
     * @tparam T The element type of the payload.
     * @param output The output payload to convert to an input.
     * @return The new input payload instance.
     */
    template <typename T>
    MPIWCPP17_INLINE payload_in_t<T> to_input(const payload_out_t<T>& output) noexcept
    {
        return payload_in_t<T>(output, output.count);
    }

    /**
     * Creates a new output payload by allocating the requested number of elements.
     * @tparam T The element type of the output payload to create.
     * @param count The total number of elements to be allocated.
     * @return The new output payload instance.
     */
    template <typename T>
    MPIWCPP17_INLINE payload_out_t<T> create_output(size_t count)
    {
        auto ptr = std::shared_ptr<T>(new T[count]);
        return payload_out_t<T>(ptr, count);
    }

    /**
     * Copies the contents of an input payload to an owning output payload.
     * @tparam T The element type of both payloads.
     * @param input The input payload to be copied.
     * @return The new output payload instance.
     */
    template <typename T>
    MPIWCPP17_INLINE payload_out_t<T> copy_to_output(const payload_in_t<T>& input)
    {
        auto ptr = std::shared_ptr<T>(new T[input.count]);
        std::copy(input.ptr, input.ptr + input.count, ptr.get());
        return payload_out_t<T>(ptr, input.count);
    }
}

MPIWCPP17_END_NAMESPACE
