/**
 * A thin C++17 wrapper for MPI.
 * @file MPI object handle for RAII automation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2025-present Rodrigo Siqueira
 */
#pragma once

#include <cstdint>
#include <utility>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/guard.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail
{
    /**
     * Generic transparent handle for raw MPI types.
     * A raw handle type should never be directly manipulated by user, but use the
     * inherited types instead. The handle automatically deletes the wrapped MPI
     * object if it owns the reference.
     * @tparam T The raw MPI type wrapped by the handle.
     * @tparam D The deleter function for the wrapped MPI type.
     * @since 2.1
     */
    template <typename T, auto D>
    class handle_t
    {
        public:
            typedef T raw_t;

        private:
            T m_handle = T {};
            bool m_owned = false;

        static_assert(sizeof(T) <= sizeof(uintmax_t) && std::is_trivially_copyable_v<T>
          , "incompatible MPI implementation: handler type is larger than a pointer");
        static_assert(std::is_same_v<decltype(D), int(*)(T*)>
          , "deleter provided to handler is incompatible with handler type");

        public:
            MPIWCPP17_INLINE handle_t() noexcept = default;
            MPIWCPP17_INLINE handle_t(const handle_t&) noexcept = delete;

            /**
             * Instantiates a new handle with the raw MPI object reference. Takes
             * ownership of the given raw handle reference if requested.
             * @param raw The raw handle reference to be wrapped.
             * @param owned Should the handle take ownership of the given object?
             */
            MPIWCPP17_INLINE handle_t(T raw, bool owned = false) noexcept
              : m_handle (raw)
              , m_owned  (owned)
            {}

            /**
             * Instantiates a new handle by moving a foreign handle instance.
             * @param other The foreign handle to move into this instance.
             */
            MPIWCPP17_INLINE handle_t(handle_t&& other) noexcept
              : m_handle (other.m_handle)
              , m_owned  (std::exchange(other.m_owned, false))
            {}

            MPIWCPP17_INLINE handle_t& operator=(const handle_t&) = delete;

            /**
             * Moves a foreign handle into this instance.
             * @param other The foreign handle to move into this instance.
             * @return This handle instance reference.
             * @see mpi::detail::handle_t::acquire
             */
            MPIWCPP17_INLINE handle_t& operator=(handle_t&& other)
            {
                acquire(std::forward<handle_t>(other)); return *this;
            }

            /**
             * Destroys the wrapped object if it is owned.
             * @see mpi::detail::handle_t::handle_t
             * @see mpi::detail::handle_t::destroy
             */
            MPIWCPP17_INLINE ~handle_t()
            {
                destroy();
            }

            /**#@+
             * The address operator of a handle.
             * This operator allows the seamless use of a handle with native MPI
             * function calls, where a pointer is needed for the wrapped MPI object.
             * @return A pointer to the wrapped MPI object.
             */
            MPIWCPP17_INLINE       T* operator&() noexcept       { return &m_handle; }
            MPIWCPP17_INLINE const T* operator&() const noexcept { return &m_handle; }
            /**#@-*/

            /**
             * The implicit converter operator of a handle.
             * This operator allows to seamlessly use a handle with native MPI function
             * calls, where the raw object is required.
             * @return The wrapped MPI object.
             */
            MPIWCPP17_INLINE operator T() const noexcept { return m_handle; }

            /**
             * The explicit boolean converter operator of a handle.
             * A handle is considered not-empty if the wrapped object is not empty.
             * @return The handle implicit boolean value.
             */
            MPIWCPP17_INLINE explicit operator bool() const noexcept
            {
                return (bool) m_handle;
            }

            /**
             * Releases ownership of the wrapped MPI object.
             * @return The wrapped MPI object.
             */
            MPIWCPP17_INLINE T release() noexcept
            {
                m_owned = false; return m_handle;
            }

        private:
            MPIWCPP17_INLINE void acquire(handle_t&&);
            MPIWCPP17_INLINE void destroy();
    };

    /**
     * Moves a foreign handle into this handle instance and acquires its ownership.
     * @tparam T The raw MPI type wrapped by the handle.
     * @tparam D The deleter function for the wrapped MPI type.
     * @param other The handle instance to be moved.
     */
    template <typename T, auto D>
    MPIWCPP17_INLINE void handle_t<T, D>::acquire(handle_t&& other)
    {
        if (this != std::addressof(other)) {
            this->destroy();
            this->m_handle = other.m_handle;
            this->m_owned  = std::exchange(other.m_owned, false);
        }
    }

    /**
     * Destroys the object wrapped by the handle if it is owned.
     * @tparam T The raw MPI type wrapped by the handle.
     * @tparam D The deleter function for the wrapped MPI type.
     */
    template <typename T, auto D>
    MPIWCPP17_INLINE void handle_t<T, D>::destroy()
    {
        if (m_handle && m_owned) {
            guard((D)(&m_handle));
            m_owned = false;
        }
    }
}

/*
 * Helper macro to quickly create inherited handle types. The inherited type does
 * not expose some methods that may be used internally by the framework.
 * @tparam T The raw MPI type wrapped by the handle.
 * @tparam D The deleter function for the wrapped MPI type.
 */
#define MPIWCPP17_INHERIT_HANDLE(T, D)          \
  public detail::handle_t<T, D> {               \
    private:                                    \
      using super_t = detail::handle_t<T, D>;   \
      using super_t::release;                   \
    public:                                     \
      using super_t::handle_t;                  \
      using super_t::operator=;                 \
  }

MPIWCPP17_END_NAMESPACE
