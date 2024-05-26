/**
 * A thin C++17 wrapper for MPI.
 * @file Internal object for deferring the destruction of resources.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <vector>

#include <mpiwcpp17/environment.h>

MPIWCPP17_BEGIN_NAMESPACE

MPIWCPP17_INLINE void abort(int);
MPIWCPP17_INLINE void finalize();

namespace detail
{
    /**
     * Defers the invokation of a callable, to the moment before MPI finalization.
     * This class is reserved for internal usage.
     * @since 3.0
     */
    class deferrer_t final
    {
        public:
            using callable_t = void();

        private:
            MPIWCPP17_INLINE static std::vector<callable_t*> s_callable;

        private:
            MPIWCPP17_CONSTEXPR deferrer_t() noexcept = default;
            MPIWCPP17_CONSTEXPR deferrer_t(const deferrer_t&) noexcept = default;
            MPIWCPP17_CONSTEXPR deferrer_t(deferrer_t&&) noexcept = default;

            MPIWCPP17_INLINE deferrer_t& operator=(const deferrer_t&) noexcept = default;
            MPIWCPP17_INLINE deferrer_t& operator=(deferrer_t&&) noexcept = default;

        public:
            /**
             * Registers a new callable to be deferred.
             * @param callable The callable to be deferred.
             */
            MPIWCPP17_INLINE explicit deferrer_t(callable_t *callable)
            {
                s_callable.push_back(callable);
            }

        private:
            /**
             * Runs the registered deferred callables.
             * @see detail::deferrer_t
             */
            MPIWCPP17_INLINE static void run()
            {
                for (callable_t *callable : s_callable)
                    (callable)();
            }

        friend void mpiwcpp17::abort(int);
        friend void mpiwcpp17::finalize();
    };
}

MPIWCPP17_END_NAMESPACE
