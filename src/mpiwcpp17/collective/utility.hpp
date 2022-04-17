/**
 * A thin C++17 wrapper for MPI.
 * @file Utility functions and structures for MPI collective operations.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <map>
#include <utility>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/datatype.hpp>
#include <mpiwcpp17/functor.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace collective::utility
{
    /**
     * Resolves an operator functor by an already created identifier.
     * @tparam T The type the operator works over.
     * @param f The given operator functor identifier.
     * @return The resolved functor identifier.
     */
    template <typename T>
    inline auto resolvef(functor::id f) noexcept -> functor::id
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
    inline auto resolvef(const F&)
    -> typename std::enable_if<
        std::is_class<F>::value &&
        std::is_default_constructible<F>::value &&
        std::is_assignable<T&, std::invoke_result_t<F, const T&, const T&>>::value
      , functor::id
    >::type
    {
        return functor::create<T,F>(false);
    }

    /**
     * Mapping between the a datatype identifier and a type-erased operator functor.
     * This is used solely for injecting operators into reduce or scan operations.
     * @since 1.0
     */
    inline static std::map<datatype::id, void*> fmap;

    /**
     * Resolves an operator functor by a callable operator function.
     * @tparam T The type the operator works over.
     * @param f The operator's raw implementation.
     * @return The resolved functor identifier.
     */
    template <typename T>
    inline auto resolvef(T (*f)(const T&, const T&)) -> functor::id
    {
        struct wrapperf {
            using F = T(const T&, const T&);

            F *lambda;
            datatype::id tid = datatype::identify<T>();

            inline wrapperf() noexcept : lambda (reinterpret_cast<F*>(fmap[tid])) {}
            inline wrapperf(F* f) noexcept { fmap[tid] = reinterpret_cast<void*>(f); }
            inline T operator()(const T& a, const T& b) { return (lambda)(a, b); }
        };

        return utility::resolvef<T>(wrapperf(f));
    }
}

MPIWCPP17_END_NAMESPACE
