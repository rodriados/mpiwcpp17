/**
 * A thin C++17 wrapper for MPI.
 * @file RAII instrumentation for instantiated MPI objects.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <utility>
#include <unordered_map>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/guard.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail
{
    /**
     * Tracks MPI objects instantiated during execution that should eventually be
     * freed before MPI finalizes. The registry maps a generic MPI object instance
     * to its MPI-destructor function to simulate RAII.
     * @since 2.1
     */
    class raii_t final
    {
        private:
            using deletefn_t = int(void*);
            using registry_t = std::unordered_map<uintptr_t, deletefn_t*>;

        private:
            MPIWCPP17_INLINE static auto s_registry = registry_t();

        public:
            /**
             * Attachs a generic MPI object to the RAII mechanism.
             * @tparam T The trackable MPI object type.
             * @tparam F The object deleter function type.
             * @param object The object to be tracked by the RAII mechanism.
             * @param deleter The function to use for object destruction.
             * @return The given unmodified object.
             */
            template <typename T, typename F>
            MPIWCPP17_INLINE static auto attach(T object, F *deleter)
            -> std::enable_if_t<std::is_function_v<F>, T> {
                s_registry.emplace(key(object), reinterpret_cast<deletefn_t*>(deleter));
                return object;
            }

            /**
             * Detaches a generic MPI object from the RAII mechanism.
             * @tparam T The trackable MPI object type.
             * @param object The object to be untracked and possibly deleted.
             * @param preserve Should the object be preserved when detaching?
             * @return Has the object been successfully detached?
             */
            template <typename T>
            MPIWCPP17_INLINE static bool detach(T object, bool preserve = false)
            {
                auto entry = s_registry.extract(key(object));
                if (!preserve && !entry.empty())
                    guard((entry.mapped())(&object));
                return !entry.empty();
            }

            /**
             * Clears the tracked objects and destroys them if requested.
             * @param preserve Should the instances be preserved when removed?
             * @see detail::raii_t::attach
             */
            MPIWCPP17_INLINE static void clear(bool preserve = false)
            {
                if (!preserve) for (auto [key, deleter] : s_registry)
                    guard((deleter)((void*)&key));
                s_registry.clear();
            }

        private:
            /**
             * Converts a generic object into a RAII-registry key.
             * @tparam T The trackable MPI object type.
             * @param object The object to be converted into a key.
             * @return The corresponding key for the given object.
             */
            template <typename T>
            MPIWCPP17_INLINE static uintptr_t key(T object)
            {
                static_assert(sizeof(uintptr_t) >= sizeof(T)
                  , "incompatible MPI implementation");
                return (uintptr_t) object;
            }
    };
}

MPIWCPP17_END_NAMESPACE
