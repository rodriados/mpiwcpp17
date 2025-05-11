/**
 * A thin C++17 wrapper for MPI.
 * @file A wrapper for MPI-enabled functor operators.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>
#include <utility>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/datatype.hpp>
#include <mpiwcpp17/functor.hpp>
#include <mpiwcpp17/flag.hpp>
#include <mpiwcpp17/guard.hpp>

#include <mpiwcpp17/detail/raii.hpp>

MPIWCPP17_BEGIN_NAMESPACE

/*
 * Auxiliary macros for implementing functions that wrap the creation of new functors.
 * The newly created functors are automatically attached to RAII.
 * @param x The functor to be acquired by a handle.
 * @param B The call block to be wrapped.
 */
#define MPIWCPP17_OP_RAII(x) functor_t ((x), true)
#define MPIWCPP17_OP_CALL(B) MPIWCPP17_OP_RAII(MPIWCPP17_GUARD_CALL(MPI_Op, B))

namespace detail::functor
{
    namespace datatype = mpiwcpp17::datatype;

    /**
     * Builds a MPI operator functor from a callable.
     * @param callable The callable to create a MPI operator functor from.
     * @param commutative Are the operator's parameters commutative?
     * @return The resolved functor identifier.
     */
    MPIWCPP17_INLINE functor_t build_from_callable(
        void (*callable)(void*, void*, int*, datatype_t::raw_t*)
      , bool commutative = false
    ) {
        return MPIWCPP17_OP_CALL(MPI_Op_create(callable, commutative, &_));
    }

    /**
     * Provides a marker for dynamically resolved functors.
     * @return The dynamic resolution marker.
     */
    MPIWCPP17_INLINE datatype::attribute_t::raw_t get_dynamic_resolution_marker()
    {
        static auto marker = raii_t::register_handle(datatype::attribute::create());
        return marker;
    }

    /**
     * Resolves an operator functor from an already known identifier.
     * @tparam T The type the operator is applied to.
     * @param f The given operator functor identifier.
     * @return The resolved functor identifier.
     */
    template <typename T>
    MPIWCPP17_INLINE functor_t::raw_t resolve(const functor_t& f)
    {
        return f;
    }

    /**
     * Resolves an operator functor from a functional type.
     * @tparam T The type the operator is applied to.
     * @tparam F The function implementation type.
     * @return The resolved functor identifier and datatype.
     */
    template <
        typename T
      , typename F
      , typename = std::enable_if_t<
            std::is_class_v<F> &&
            std::is_default_constructible_v<F> &&
            std::is_assignable_v<T&, std::invoke_result_t<F, const T&, const T&>>>>
    MPIWCPP17_INLINE functor_t::raw_t resolve(const F&)
    {
        // As a convenience for the caller, a static wrapper for the operator is
        // provided so that it is adapted to the function signature required by MPI.
        // The wrapper is responsible for only correctly casting the operands and
        // applying the provided operator functor.
        using static_wrapper_t = struct static_wrapper_t {
            static void call(void *a, void *b, int *count, datatype_t::raw_t*) {
                auto f = F();
                auto x = static_cast<T*>(a);
                auto y = static_cast<T*>(b);

                for (int i = 0; i < *count; ++i, ++x, ++y)
                    *y = (f)(*x, *y);
            }
        };

        static functor_t f = raii_t::register_handle(
            build_from_callable(
                &static_wrapper_t::call
              , std::is_base_of_v<flag::functor::commutative_t, F>));

        return f;
    }

    /**
     * Resolves an operator functor from a callable function.
     * @tparam T The type the operator is applied to.
     * @tparam F The function implementation type.
     * @param lambda The callable function pointer.
     * @return The resolved functor identifier and datatype.
     */
    template <
        typename T
      , typename F
      , typename = std::enable_if_t<
            !std::is_default_constructible_v<F> &&
            std::is_assignable_v<T&, std::invoke_result_t<F, const T&, const T&>>>>
    MPIWCPP17_INLINE functor_t::raw_t resolve(F& lambda)
    {
        // As a convenience for the caller, a dynamic wrapper for the operator is
        // provided when the given operator requires a non-trivial instance for
        // execution or when it's a plain function. The wrapper is responsible for
        // adapting the operator to the function signature requested by MPI, retrieving
        // the operator and applying it to correctly-typed operands.
        using dynamic_wrapper_t = struct dynamic_wrapper_t {
            static void call(void *a, void *b, int *count, datatype_t::raw_t *marker) {
                auto [ok, f] = datatype::attribute::get<F>(*marker, get_dynamic_resolution_marker());
                auto x = static_cast<T*>(a);
                auto y = static_cast<T*>(b);

                if (ok) for (int i = 0; i < *count; ++i, ++x, ++y)
                    *y = (*f)(*x, *y);
            }
        };

        auto type = datatype::identify<T>();
        auto marker = get_dynamic_resolution_marker();

        // Attention! The lambda reference is not owned by the wrapper, therefore
        // the caller is responsible for guaranteeing that it exists throughout the
        // lifetime of the operation under execution.
        datatype::attribute::set(type, marker, &lambda);

        static functor_t f = raii_t::register_handle(
            build_from_callable(&dynamic_wrapper_t::call));

        return f;
    }
}

#undef MPIWCPP17_OP_CALL
#undef MPIWCPP17_OP_RAII

MPIWCPP17_END_NAMESPACE
