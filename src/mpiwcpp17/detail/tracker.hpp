/**
 * A thin C++17 wrapper for MPI.
 * @file Tracker for instantiated MPI execution objects.
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
     * freed before MPI finalizes. The tracker maps a generic object instance to
     * its MPI-destructor function.
     * @since 3.0
     */
    class tracker_t final
    {
        private:
            using deleter_t = int(*)(void*);

        private:
            MPIWCPP17_INLINE static auto s_map = std::unordered_map<uintptr_t, deleter_t*>();

        public:
            /**
             * Adds a generic instance to the tracker.
             * @tparam T The trackable instance type.
             * @tparam F The instance deleter function type.
             * @param instance The instance to be tracked.
             * @param deleter The function to use for instance destruction.
             * @return The tracked instance.
             */
            template <typename T, typename F>
            MPIWCPP17_INLINE static auto add(T instance, F deleter) -> T
            {
                s_map.try_emplace(
                    (uintptr_t) instance
                  , reinterpret_cast<deleter_t*>(deleter));

                return instance;
            }

            /**
             * Removes an instance from the tracker and destroys it if requested.
             * @tparam T The trackable instance type.
             * @param instance The instance to be removed.
             * @param preserve Should the instance be preserved when removed?
             * @return Has the instance been removed from the tracker?
             */
            template <typename T>
            MPIWCPP17_INLINE static bool remove(T instance, bool preserve = false)
            {
                auto target = (uintptr_t) instance;
                auto obj    = s_map.extract(target);

                if (!preserve && !obj.empty())
                    guard(reinterpret_cast<deleter_t>(obj.mapped())(&instance));

                return obj.empty();
            }

            /**
             * Clears the tracker and removes all tracked instances.
             * @param preserve Should the instances be preserved when removed?
             * @see detail::tracker_t::add
             */
            MPIWCPP17_INLINE static void clear(bool preserve = false)
            {
                if (!preserve) for (auto [key, deleter] : s_map)
                    guard(reinterpret_cast<deleter_t>(deleter)((void*)&key));

                s_map.clear();
            }
    };
}

MPIWCPP17_END_NAMESPACE
