/**
 * A thin C++17 wrapper for MPI.
 * @file MPI type-independent message payload for collective operations.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <array>
#include <memory>
#include <variant>
#include <utility>
#include <iterator>
#include <algorithm>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/detail/container.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail
{
    /**
     * A generic storage buffer for payload data to transit through MPI collective
     * operations. The buffer makes the best effort to carry as many elements as
     * possible inplace, avoiding unnecessary memory allocation.
     * @tparam T The payload message buffer type.
     * @tparam N The number of buffer elements to carry inplace.
     * @since 2.1
     */
    template <typename T, size_t N = std::max(1UL, sizeof(void*) / sizeof(T))>
    struct payload_storage_t
    {
        MPIWCPP17_CONSTEXPR static size_t local_storage_threshold = N;

        using dynamic_storage_t = std::unique_ptr<T[]>;
        using local_storage_t = std::array<T, local_storage_threshold>;
        using storage_t = std::variant<dynamic_storage_t, local_storage_t>;

        storage_t storage;

        MPIWCPP17_INLINE payload_storage_t() = default;
        MPIWCPP17_INLINE payload_storage_t(const payload_storage_t&) = delete;
        MPIWCPP17_INLINE payload_storage_t(payload_storage_t&&) = default;

        /**
         * Initializes a new payload storage buffer with the given number of elements.
         * @param count The total number of elements to allocate storage for.
         */
        MPIWCPP17_INLINE payload_storage_t(size_t count)
          : storage (
                count > local_storage_threshold
                  ? storage_t(dynamic_storage_t(new T[count]))
                  : storage_t(local_storage_t())
            ) {}

        MPIWCPP17_INLINE payload_storage_t& operator=(const payload_storage_t&) = delete;
        MPIWCPP17_INLINE payload_storage_t& operator=(payload_storage_t&&) = default;

        /**
         * Exposes the owned payload storage buffer.
         * @return The pointer to the payload storage buffer.
         */
        MPIWCPP17_INLINE T *get_storage_buffer()
        {
            constexpr auto f = [](auto&& buffer) -> T& { return buffer[0]; };
            return std::addressof(std::visit(f, this->storage));
        }
    };

    /**
     * A message payload of input data to send through a MPI collective operation.
     * In practice, the input payload wraps an outgoing message in a neutral context
     * for transmission and does not own the message buffer.
     * @tparam The payload message buffer type.
     * @since 2.0
     */
    template <typename T>
    struct payload_in_t : public container_t<T>
    {
        MPIWCPP17_CONSTEXPR payload_in_t() noexcept = default;
        MPIWCPP17_CONSTEXPR payload_in_t(const payload_in_t&) noexcept = default;
        MPIWCPP17_CONSTEXPR payload_in_t(payload_in_t&&) noexcept = default;

        /**
         * Initializes a new input payload from a raw pointer.
         * @param ptr The pointer to wrap as input.
         * @param count The total number of elements to carry as input.
         */
        MPIWCPP17_CONSTEXPR payload_in_t(T *ptr, size_t count = 1) noexcept
          : container_t<T> (ptr, count)
        {}

        /**
         * Initializes a new input from a pair of pointer and element count.
         * @param input The pair containing a pointer and respective element count.
         */
        MPIWCPP17_CONSTEXPR payload_in_t(const std::pair<T*, size_t>& input) noexcept
          : payload_in_t (input.first, input.second)
        {}

        MPIWCPP17_INLINE payload_in_t& operator=(const payload_in_t&) = default;
        MPIWCPP17_INLINE payload_in_t& operator=(payload_in_t&&) = default;
    };

    /**
     * A message payload of output data received through a MPI collective operation.
     * In practice, this serves as an ownership data container for incoming messages.
     * @tparam T The payload message buffer type.
     * @since 2.0
     */
    template <typename T>
    struct payload_out_t : private payload_storage_t<T>, public container_t<T>
    {
        MPIWCPP17_INLINE payload_out_t() noexcept = default;
        MPIWCPP17_INLINE payload_out_t(const payload_out_t&) noexcept = delete;

        /**
         * Initializes a new output payload with the given number of elements.
         * @param count The total number of elements to allocate as output.
         */
        MPIWCPP17_INLINE payload_out_t(size_t count)
          : payload_storage_t<T> (count)
          , container_t<T> (this->get_storage_buffer(), count)
        {}

        /**
         * Initializes a new output by moving an already allocated payload.
         * @param other The payload instance to be moved.
         */
        MPIWCPP17_INLINE payload_out_t(payload_out_t&& other)
          : payload_storage_t<T> (std::forward<payload_out_t>(other))
          , container_t<T> (this->get_storage_buffer(), other.count)
        {}

        MPIWCPP17_INLINE payload_out_t& operator=(const payload_out_t&) = delete;
        MPIWCPP17_INLINE payload_out_t& operator=(payload_out_t&&) = delete;

        /**
         * Implicitly converts an output payload into an input payload for inplace
         * operations. All MPI operations are inplace, so only input payloads can
         * directly interact with native functions. Input payloads, on the other
         * hand, do not own their buffers, therefore we must create it from output.
         * @return The implicitly converted input payload.
         */
        MPIWCPP17_INLINE operator payload_in_t<T>() const noexcept
        {
            return payload_in_t(this->ptr, this->count);
        }
    };
}

namespace detail::payload
{
    /**
     * Determines whether a given type is contiguously iterable, therefore being
     * able to be converted as a container into an input payload.
     * @tparam T The type to test if it is a contiguous iterable.
     * @since 2.0
     */
    template <typename T, typename = void>
    MPIWCPP17_CONSTEXPR bool is_contiguous_iterable_v = false;

    template <typename T>
    MPIWCPP17_CONSTEXPR bool is_contiguous_iterable_v<T, std::enable_if_t<
        std::is_base_of_v<
            std::random_access_iterator_tag
          , typename std::iterator_traits<decltype(std::begin(std::declval<T&>()))>
                ::iterator_category>>> = true;

    /**
     * Tentatively transforms a container of contiguous elements into an input payload.
     * @tparam C The type of contiguous memory container to be wrapped.
     * @param container The container instance to be wrapped.
     * @return The buffer pointer and count of given container.
     */
    template <typename C>
    MPIWCPP17_INLINE auto to_tentative_input(
        C& container
      , std::enable_if_t<is_contiguous_iterable_v<C>>* = nullptr
    ) {
        return std::make_pair(
            std::addressof(*std::begin(container))
          , (size_t) std::distance(std::begin(container), std::end(container))
        );
    }

    /**
     * Tentatively transforms a plain data object into an input payload.
     * @tparam T The input object type for the payload.
     * @param data The object to be carried by the payload.
     * @return The pointer and count of the given data object.
     */
    template <typename T>
    MPIWCPP17_INLINE auto to_tentative_input(
        T& data
      , std::enable_if_t<!is_contiguous_iterable_v<T>>* = nullptr
    ) noexcept {
        return std::make_pair(&data, size_t(1));
    }

    /**
     * Tentatively transforms an output payload into an input payload.
     * @tparam T The output payload element type.
     * @param output The output payload to convert into input.
     * @return The pointer and count of the given output payload.
     */
    template <typename T>
    MPIWCPP17_INLINE auto to_tentative_input(const payload_out_t<T>& output) noexcept
    {
        return std::make_pair(output, output.count);
    }

    /**
     * Creates a new input payload from a generic object.
     * @tparam G The generic type to create an input from.
     * @param input The generic object to convert into an input payload.
     * @return The new input payload instance.
     */
    template <typename G>
    MPIWCPP17_INLINE auto to_input(G&& input)
    {
        return payload_in_t(to_tentative_input(std::forward<G>(input)));
    }

    /**
     * Creates a new output payload by allocating the requested number of elements.
     * @tparam T The element type of the output payload to create.
     * @param count The total number of elements to be allocated.
     * @return The new output payload instance.
     */
    template <typename T>
    MPIWCPP17_INLINE auto create_output(size_t count)
    {
        return payload_out_t<std::remove_cv_t<T>>(count);
    }

    /**
     * Copies the contents of an input payload to an owning output payload.
     * @tparam T The element type of both payloads.
     * @param input The input payload to be copied.
     * @return The new output payload instance.
     */
    template <typename T>
    MPIWCPP17_INLINE auto copy_to_output(const payload_in_t<T>& input)
    {
        auto output = payload_out_t<std::remove_cv_t<T>>(input.count);
        std::copy(input.begin(), input.end(), output.begin());
        return output;
    }
}

MPIWCPP17_END_NAMESPACE
