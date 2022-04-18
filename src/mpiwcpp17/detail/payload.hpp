/**
 * A thin C++17 wrapper for MPI.
 * @file MPI type-independent MPI message payload for collective operations.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <utility>
#include <memory>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/datatype.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail
{
    /**
     * An incoming or outgoing message payload to transit through a MPI collective
     * operation between different processes. In practice, this serves as a neutral,
     * commom context for messages of different types.
     * @tparam T The payload's message type.
     * @since 1.0
     */
    template <typename T, typename = void>
    struct payload;

    /**
     * A payload of generic non-union and trivial type.
     * @tparam T The payload's message type.
     * @since 1.0
     */
    template <typename T>
    struct payload<T, typename std::enable_if<!std::is_union<T>() && std::is_trivial<T>()>::type>
    {
        typedef T element_type;
        typedef element_type *pointer_type;

        using return_type = payload<T>;

        std::shared_ptr<element_type[]> ptr;
        datatype::id type = datatype::identify<element_type>();
        size_t count = 0;

        inline payload() = default;
        inline payload(const payload&) noexcept = default;
        inline payload(payload&&) noexcept = default;

        /**
         * Wraps a simple value into a payload instance.
         * @param value The payload's message simple contents.
         */
        inline payload(element_type& value) noexcept
          : payload (&value, 1)
        {}

        /**
         * Wraps a raw non-owning pointer into a message payload.
         * @param ptr The raw message buffer pointer.
         * @param count The total number of elements in message.
         */
        inline payload(pointer_type ptr, size_t count = 1) noexcept
          : ptr (ptr, [](auto) { /* pointer is not owned */ })
          , count (count)
        {}

        /**
         * Wraps an owning pointer into a message payload.
         * @tparam U The original owning shared pointer type.
         * @param ptr The raw message owning pointer.
         * @param count The total number of elements in message.
         */
        template <typename U>
        inline payload(const std::shared_ptr<U>& ptr, size_t count = 1) noexcept
          : ptr (std::static_pointer_cast<element_type[]>(ptr))
          , count (count)
        {}

        inline payload& operator=(const payload&) = default;
        inline payload& operator=(payload&&) = default;

        /**
         * The payload's message buffer initial iterator position.
         * @return The pointer to the start of the message.
         */
        inline pointer_type begin() noexcept
        {
            return ptr.get();
        }

        /**
         * The payload's message buffer initial const-qualified iterator position.
         * @return The pointer to the start of the const-qualified message.
         */
        inline const pointer_type begin() const noexcept
        {
            return ptr.get();
        }

        /**
         * The payload's message final iterator position.
         * @return The pointer to the end of the message.
         */
        inline pointer_type end() noexcept
        {
            return ptr.get() + count;
        }

        /**
         * The payload's message final const-qualified iterator position.
         * @return The pointer to the end of the const-qualified message.
         */
        inline const pointer_type end() const noexcept
        {
            return ptr.get() + count;
        }

        /**
         * Seamlessly converts the payload into its message contents.
         * @return The payload's message contents.
         */
        inline operator element_type&() noexcept
        {
            return *ptr.get();
        }

        /**
         * Seamlessly converts the payload into its const-qualified message contents.
         * @return The payload's const-qualified message contents.
         */
        inline operator const element_type&() const noexcept
        {
            return *ptr.get();
        }

        /**
         * Converts the payload into a pointer to its message contents.
         * @return The pointer to the payload's message contents.
         */
        inline operator pointer_type() noexcept
        {
            return ptr.get();
        }

        /**
         * Converts the payload into a const-qualified pointer to its message contents.
         * @return The const-qualified pointer to the payload's message contents.
         */
        inline operator const pointer_type() const noexcept
        {
            return ptr.get();
        }

        /**
         * Creates a new payload by allocating a message of given number of elements.
         * @param count The total amount of elements in the message to be allocated.
         * @return The newly created payload.
         */
        inline static payload create(size_t count = 1)
        {
            return {std::shared_ptr<element_type[]>(new element_type[count]), count};
        }
    };

    /**
     * A payload context for messages created using iterable types.
     * @tparam C The iterable container type.
     * @tparam T The container's content type.
     * @since 1.0
     */
    template <template <typename> class C, typename T>
    struct payload<C<T>, std::void_t<
        decltype(std::declval<C<T>&>().begin())
      , decltype(std::declval<C<T>&>().end())
    >> : public payload<T>
    {
        /**
         * Initializes a payload from the container instance.
         * @param container The container to create a message payload from.
         */
        inline payload(C<T>& container) noexcept(noexcept(container.begin()) && noexcept(container.end()))
          : payload<T> (&(*container.begin()), static_cast<size_t>(&(*container.end()) - &(*container.begin())))
        {}

        using payload<T>::operator=;
    };

    /*
     * Deduction guides for payloads.
     * @since 1.0
     */
    template <typename T> payload(T*, size_t = 1) -> payload<T>;
    template <typename T> payload(T&, size_t = 1) -> payload<T>;
    template <typename T> payload(std::shared_ptr<T>&, size_t = 1) -> payload<T>;
    template <typename T> payload(std::shared_ptr<T[]>&, size_t = 1) -> payload<T>;
}

MPIWCPP17_END_NAMESPACE
