/**
 * A thin C++17 wrapper for MPI.
 * @file Utility functions and structures for MPI collective operations.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <utility>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/functor.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace detail::collective
{
    /**
     * Resolves an operator functor by an already created identifier.
     * @tparam T The type the operator works over.
     * @param f The given operator functor identifier.
     * @return The resolved functor identifier.
     */
    template <typename T>
    inline auto resolve_functor(functor::raw_t f) noexcept -> functor::raw_t
    {
        return f;
    }

    /**
     * Resolves an operator functor by an operator implementation type.
     * @tparam T The type the operator works over.
     * @tparam F The operator implementation type.
     * @return The resolved functor identifier.
     */
    template <typename T, typename F>
    inline auto resolve_functor(const F&)
    -> typename std::enable_if<
        std::is_class<F>::value &&
        std::is_default_constructible<F>::value &&
        std::is_invocable_r<T, F, const T&, const T&>::value
      , functor::raw_t
    >::type
    {
        return functor::create<T, F>(false);
    }

    /**
     * Resolves an operator functor by a callable operator function.
     * @tparam T The type the operator works over.
     * @tparam F The operator implementation type.
     * @param f The operator's raw implementation.
     * @return The resolved functor identifier.
     */
    template <typename T, typename F>
    inline auto resolve_functor(const F& f)
    -> typename std::enable_if<
        !std::is_default_constructible<F>::value &&
        std::is_invocable_r<T, F, const T&, const T&>::value
      , functor::raw_t
    >::type
    {
        static std::conditional_t<std::is_function_v<F>, F*, F> lambda = f;

        using fwrapper_t = struct {
            inline T operator()(const T& a, const T& b) {
                return (lambda)(a, b);
            }
        };

        (void) new (&lambda) decltype(lambda) {f};
        return detail::collective::resolve_functor<T>(fwrapper_t());
    }
}

MPIWCPP17_END_NAMESPACE
