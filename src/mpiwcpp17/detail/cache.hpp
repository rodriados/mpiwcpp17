/**
 * A thin C++17 wrapper for MPI.
 * @file Cache of instantiated MPI execution objects.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2025-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <utility>
#include <unordered_map>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/guard.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail::cache
{
    /**
     * Configures how a type must interact with the cache buffer. Every specialization
     * must provide static methods to hash and destroy an instance of the type.
     * @tparam T The cacheable type to be configured.
     * @since 3.0
     */
    template <
        typename T
      , typename = std::enable_if_t<std::is_pointer_v<T> || std::is_integral_v<T>>
    >
    struct configuration_t;
}

/**
 * Configures the cache for instances of the given type.
 * @param type The type to configure the cache for.
 * @param fn The destructor function for an instance of the given type.
 */
#define MPIWCPP17_CACHE_CONFIGURE(type, fn)                                    \
    template <> struct mpiwcpp17::detail::cache::configuration_t<type> {       \
        MPIWCPP17_INLINE static uintptr_t hash(const type& instance) {         \
            return reinterpret_cast<uintptr_t>(instance);                      \
        }                                                                      \
        MPIWCPP17_INLINE static void destroy(void *instance) {                 \
            type reference = static_cast<type>(instance);                      \
            mpiwcpp17::guard((fn)(&reference));                                \
        }                                                                      \
    }

namespace detail::cache
{
    /**
     * The cache buffer which contains references to every stored instance. The
     * cache is not necessarily used for optimizations but also to delete objects
     * before MPI is finalized.
     * @since 3.0
     */
    MPIWCPP17_INLINE static auto buffer = std::unordered_map<uintptr_t, void(*)(void*)>();

    /**
     * Adds an instance to the cache.
     * @tparam T The cacheble instance type.
     * @param instance The instance to be cached.
     * @return The cached instance.
     */
    template <typename T>
    MPIWCPP17_INLINE auto add(const T& instance) -> const T&
    {
        using C = configuration_t<T>;
        buffer.insert({C::hash(instance), &C::destroy});
        return instance;
    }

    /**
     * Removes an instance from the cache and destroys it.
     * @tparam T The cacheable instance type.
     * @param instance The instance to be destroyed.
     * @return Has the instance been destroyed?
     */
    template <typename T>
    MPIWCPP17_INLINE bool remove(const T& instance)
    {
        using C = configuration_t<T>;
        auto cached = buffer.extract(C::hash(instance));
        if (!cached.empty()) { (cached.mapped()) (reinterpret_cast<void*>(cached.key())); }
        return !cached.empty();
    }

    /**
     * Clears the cache and destroys all instances.
     * @see detail::cache::add
     */
    MPIWCPP17_INLINE void clear()
    {
        for (auto& cached : buffer)
            (cached.second) (reinterpret_cast<void*>(cached.first));
        buffer.clear();
    }
}

MPIWCPP17_END_NAMESPACE
