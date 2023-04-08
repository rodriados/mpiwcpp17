/**
 * A thin C++17 wrapper for MPI.
 * @file MPI type-independent MPI message payload for collective operations.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <memory>
#include <utility>
#include <iterator>
#include <algorithm>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/datatype.hpp>

MPIWCPP17_BEGIN_NAMESPACE

/**
 * An incoming or outgoing message payload to transit through a MPI collective
 * operation between different processes. In practice, this serves as a neutral,
 * commom context for messages of different types.
 * @tparam T The payload's message type.
 * @since 1.0
 */
template <typename T, typename = void>
struct payload_t;

/**
 * A payload of generic non-union and trivial type.
 * @tparam T The payload's message type.
 * @since 1.0
 */
template <typename T>
struct payload_t<T, typename std::enable_if<!std::is_union<T>() && std::is_trivially_copyable<T>()>::type>
{
    typedef T element_t;
    typedef element_t *pointer_t;

    using return_t = payload_t<T>;

    std::shared_ptr<element_t[]> ptr;
    datatype_t type = datatype::identify<element_t>();
    size_t count = 0;

    inline payload_t() = default;
    inline payload_t(const payload_t&) noexcept = default;
    inline payload_t(payload_t&&) noexcept = default;

    /**
     * Wraps a simple value into a payload instance.
     * @param value The payload's message simple contents.
     */
    inline payload_t(element_t& value) noexcept
      : payload_t (&value, 1)
    {}

    /**
     * Wraps a raw non-owning pointer into a message payload.
     * @param ptr The raw message buffer pointer.
     * @param count The total number of elements in message.
     */
    inline payload_t(pointer_t ptr, size_t count = 1) noexcept
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
    inline payload_t(const std::shared_ptr<U>& ptr, size_t count = 1) noexcept
      : ptr (std::static_pointer_cast<element_t[]>(ptr))
      , count (count)
    {}

    /**
     * Initializes a new payload from a list of its elements.
     * @param list The list with the elements of the payload.
     */
    inline payload_t(const std::initializer_list<element_t>& list)
      : ptr (new element_t[list.size()])
      , count (list.size())
    {
        std::copy(list.begin(), list.end(), ptr.get());
    }

    inline payload_t& operator=(const payload_t&) = default;
    inline payload_t& operator=(payload_t&&) = default;

    /**
     * The payload's message buffer initial iterator position.
     * @return The pointer to the start of the message.
     */
    inline pointer_t begin() noexcept
    {
        return ptr.get();
    }

    /**
     * The payload's message buffer initial const-qualified iterator position.
     * @return The pointer to the start of the const-qualified message.
     */
    inline const pointer_t begin() const noexcept
    {
        return ptr.get();
    }

    /**
     * The payload's message final iterator position.
     * @return The pointer to the end of the message.
     */
    inline pointer_t end() noexcept
    {
        return ptr.get() + count;
    }

    /**
     * The payload's message final const-qualified iterator position.
     * @return The pointer to the end of the const-qualified message.
     */
    inline const pointer_t end() const noexcept
    {
        return ptr.get() + count;
    }

    /**
     * Exposes the payload's contents by an index.
     * @param index The index of the element to be accessed.
     * @return A reference to the element at the given index.
     */
    inline element_t& operator[](ptrdiff_t index) const
    {
        return ptr[index];
    }

    /**
     * Seamlessly converts the payload into its message contents.
     * @return The payload's message contents.
     */
    inline operator element_t&() noexcept
    {
        return *ptr.get();
    }

    /**
     * Seamlessly converts the payload into its const-qualified message contents.
     * @return The payload's const-qualified message contents.
     */
    inline operator const element_t&() const noexcept
    {
        return *ptr.get();
    }

    /**
     * Converts the payload into a pointer to its message contents.
     * @return The pointer to the payload's message contents.
     */
    inline operator pointer_t() noexcept
    {
        return ptr.get();
    }

    /**
     * Converts the payload into a const-qualified pointer to its message contents.
     * @return The const-qualified pointer to the payload's message contents.
     */
    inline operator const pointer_t() const noexcept
    {
        return ptr.get();
    }
};

/**
 * A payload context for messages created using iterable types.
 * @tparam C The iterable contiguous container type.
 * @tparam T The container's content type.
 * @since 1.0
 */
template <template <typename> class C, typename T>
struct payload_t<C<T>, typename std::enable_if<
    std::is_same<
        typename std::iterator_traits<decltype(std::declval<C<T>&>().begin())>::iterator_category
      , std::random_access_iterator_tag
    >::value
>::type> : public payload_t<T>
{
    /**
     * Initializes a payload from the contiguous container instance.
     * @param container The contiguous container to create a message payload from.
     */
    inline payload_t(C<T>& container)
      : payload_t<T> (
            &(*container.begin())
          , std::distance(container.begin(), container.end())
        )
    {}

    using payload_t<T>::operator=;
};

/*
 * Deduction guides for payloads.
 * @since 1.0
 */
template <typename T> payload_t(const T&) -> payload_t<T>;
template <typename T> payload_t(T*, size_t = 1) -> payload_t<T>;
template <typename T> payload_t(const std::shared_ptr<T>&, size_t = 1) -> payload_t<T>;
template <typename T> payload_t(const std::shared_ptr<T[]>&, size_t = 1) -> payload_t<T>;
template <typename ...T> payload_t(T...) -> payload_t<typename std::common_type<T...>::type>;

namespace payload
{
    /**
     * Creates a new payload by allocating a message of given number of elements.
     * @tparam T The payload's message type.
     * @param count The total amount of elements in the message to be allocated.
     * @return The newly created payload.
     */
    template <typename T>
    inline typename payload_t<T>::return_t create(size_t count = 1)
    {
        using element_t = typename payload_t<T>::return_t::element_t;
        return payload_t(std::shared_ptr<element_t[]>(new element_t[count]), count);
    }
}

MPIWCPP17_END_NAMESPACE
