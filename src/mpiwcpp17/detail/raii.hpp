/**
 * A thin C++17 wrapper for MPI.
 * @file RAII instrumentation for instantiated MPI objects.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <stack>
#include <cstdint>
#include <utility>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/guard.hpp>

#include <mpiwcpp17/detail/handle.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail
{
    /**
     * Tracks static MPI objects instantiated during execution that should be destroyed
     * before MPI finalizes. The registry maps a generic MPI object handle to its
     * MPI-destructor function to achieve RAII for static object handles.
     * @since 2.1
     */
    class raii_t
    {
        private:
            class registry_entry_t;
            using registry_t = std::stack<registry_entry_t>;

        private:
            static registry_t s_registry;

        public:
            /**
             * Registers a generic MPI object handle to the RAII context.
             * @tparam T The trackable MPI object handle type.
             * @param handle The object handle to be tracked by the RAII context.
             * @return The non-owning object handle.
             */
            template <typename T>
            MPIWCPP17_INLINE static T register_handle(T&&);

            /**
             * Destroys all registered object handles in the RAII context, guarantees
             * that all handles are destroyed in their opposite registration order.
             * @see mpi::detail::raii_t::register_handle
             */
            MPIWCPP17_INLINE static void finalize();
    };

    /**
     * The static RAII registry entry for a handle of generic type.
     * The entry acquires ownership of a generic handle and its deleter, attaches
     * it to the RAII registry and destroys the handle when the registry is finalized.
     * @since 2.1
     */
    class raii_t::registry_entry_t
    {
        private:
            using deleter_t = int(*)(void*);

        private:
            uintmax_t m_handle;
            deleter_t m_deleter;

        public:
            /**
             * Instantiates a new registry entry from a generic handle.
             * @tparam T The raw MPI object handle type.
             * @tparam D The deleter function for the given handle type.
             * @param handle The handle to be owned by the registry.
             */
            template <typename T, auto D>
            MPIWCPP17_INLINE registry_entry_t(handle_t<T, D>&&) noexcept;

            /**
             * Destroys the owned handle using its extracted deleter function.
             * The deleter is expected to perform a MPI operation to release the
             * resources owned by the handle. Therefore, the deleter call is guarded
             * and may throw an exception if an error is detected.
             * @see mpi::detail::raii_t::finalize
             */
            MPIWCPP17_INLINE ~registry_entry_t();

        private:
            /**
             * Erases the type of a generic raw MPI object handle.
             * @tparam T The original raw MPI object handle type.
             * @param handle The handle to have its type erased.
             */
            template <typename T>
            MPIWCPP17_CONSTEXPR static uintmax_t to_type_erased_handle(T) noexcept;
    };

    /*
     * Initialization of RAII's internal stack registry.
     * Handles are kept in the stack throughout MPI execution and destroyed in the
     * opposite order that they are created.
     */
    MPIWCPP17_INLINE raii_t::registry_t raii_t::s_registry;

    /**
     * Registers a generic MPI object handle to the RAII context.
     * @tparam T The trackable MPI object handle type.
     * @param handle The object handle to be tracked by the RAII context.
     * @return The non-owning object handle.
     */
    template <typename T>
    MPIWCPP17_INLINE T raii_t::register_handle(T&& handle)
    {
        s_registry.emplace(std::forward<T>(handle));
        return std::move(handle);
    }

    /**
     * Instantiates a new registry entry from a generic handle.
     * @tparam T The raw MPI object handle type.
     * @tparam D The deleter function for the given handle type.
     * @param handle The handle to be owned by the registry.
     */
    template <typename T, auto D>
    MPIWCPP17_INLINE raii_t::registry_entry_t::registry_entry_t(handle_t<T, D>&& handle) noexcept
      : m_handle (to_type_erased_handle(handle.release()))
      , m_deleter (reinterpret_cast<deleter_t>(D))
    {}

    /**
     * Destroys all registered object handles in the RAII context, guarantees
     * that all handles are destroyed in their opposite registration order.
     * @see mpi::detail::raii_t::register_handle
     */
    MPIWCPP17_INLINE void raii_t::finalize()
    {
        while (!s_registry.empty()) {
            s_registry.pop();
        }
    }

    /**
     * Destroys the owned handle using its extracted deleter function.
     * The deleter is expected to perform a MPI operation to release the
     * resources owned by the handle. Therefore, the deleter call is guarded
     * and may throw an exception if an error is detected.
     * @see mpi::detail::raii_t::finalize
     */
    MPIWCPP17_INLINE raii_t::registry_entry_t::~registry_entry_t()
    {
        if (m_handle && m_deleter) {
            guard((m_deleter)(&m_handle));
        }
    }

    /**
     * Erases the type of a generic raw MPI object handle.
     * @tparam T The original raw MPI object handle type.
     * @param handle The handle to have its type erased.
     */
    template <typename T>
    MPIWCPP17_CONSTEXPR uintmax_t raii_t::registry_entry_t::to_type_erased_handle(
        T handle
    ) noexcept {
        union type_eraser_t { T handle; uintmax_t erased; };
        const auto converter = type_eraser_t {handle};
        return converter.erased;
    }
}

MPIWCPP17_END_NAMESPACE
