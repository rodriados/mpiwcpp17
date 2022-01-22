/**
 * A thin C++17 wrapper for MPI.
 * @file Reflection implementation for simple data structures.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Alexandr Poltavsky, Antony Polukhin, Rodrigo Siqueira
 */
#pragma once

/*
 * The Great Type Loophole (C++14)
 * Initial implementation by Alexandr Poltavsky, http://alexpolt.github.io
 * With participation of Antony Polukhin, http://github.com/apolukhin
 *
 * The Great Type Loophole is a technique that allows to exchange type information
 * with template instantiations. Basically you can assign and read type information
 * during compile time. Here it is used to detect data members of a data type. I
 * described it for the first time in this blog post http://alexpolt.github.io/type-loophole.html .
 *
 * This technique exploits the http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_active.html#2118
 * CWG 2118. Stateful metaprogramming via friend injection
 * Note: CWG agreed that such techniques should be ill-formed, although the mechanism
 * for prohibiting them is as yet undetermined.
 */
#include <cstddef>
#include <cstdint>
#include <utility>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/detail/tuple.hpp>

#if !defined(MPIWCPP17_AVOID_REFLECTION)

/*
 * As we are exploiting some "obscure" behaviors of the language, and using some
 * tricks that upset compilers, we need to disable some warnings in order to force
 * the compilation to take place without any warnings.
 */
MPIWCPP17_DISABLE_GCC_WARNING_BEGIN("-Wnon-template-friend")

MPIWCPP17_BEGIN_NAMESPACE

namespace detail::reflection
{
    /**
     * Reflects over the target data type, that is it iterates over and extracts
     * information about the member properties of the target type.
     * @tparam T The target data type to be introspected.
     * @since 1.0
     */
    template <typename T>
    class reflector;

    /**
     * Tags a member property type to an index for overload resolution.
     * @tparam T The target type for reflexive-introspection.
     * @tparam N The index of extracted property member type.
     * @since 1.0
     */
    template <typename T, size_t N>
    struct tag
    {
        friend auto latch(tag<T, N>) noexcept;
    };

    /**
     * Injects a friend function to couple a property type to an index.
     * @tparam T The target type for reflexive-introspection.
     * @tparam U The extracted property type.
     * @param N The index of extracted property member type.
     * @since 1.0
     */
    template <typename T, typename U, size_t N>
    struct injector
    {
        /**
         * Binds the extracted member type to its index within the target reflection
         * type. This function does not aim to have its concrete return value used,
         * but only its return type.
         * @return The extracted type bound to the member index.
         */
        friend inline auto latch(tag<T, N>) noexcept
        {
            return typename std::remove_all_extents<U>::type {};
        }
    };

    /**
     * A decoy type responsible for morphing itself into the type of a member of
     * the reflection target, and indexing and latching the extracted type into
     * the injector, so that it can be retrieved later on.
     * @tparam T The target type for reflexive-introspection.
     * @tparam N The index of property member type to extract.
     * @since 1.0
     */
    template <typename T, size_t N>
    struct decoy
    {
        /**
         * Injects and links an extracted member type into a latch, with its index,
         * if it has not yet been previously done so.
         * @tparam U The extracted property member type.
         * @tparam I The index of the property member being analyzed.
         */
        template <typename U, size_t I>
        static constexpr auto inject(...) -> injector<T, U, I>;

        /**
         * Validates whether the property member under analysis has already been
         * extracted and reflected over. If yes, avoids latch redeclaration.
         * @tparam I The index of the property member being processed.
         */
        template <typename, size_t I>
        static constexpr auto inject(int) -> decltype(latch(tag<T, I>()));

        /**
         * Morphs the decoy into the type of a property member of the target reflection
         * type, and injects it into a latch.
         * @tparam U The type to morph the decoy into.
         */
        template <typename U, size_t = sizeof(inject<U, N>(0))>
        constexpr operator U&() const;
    };

    /**#@+
     * Counts the number of property members in the target type by increasingly
     * injecting parameter values to its constructor, using SFINAE. This is the
     * step within a type's reflection processing, that requires it to be a trivial
     * data type. If a custom constructor is defined, then the constructor parameter
     * types will be extracted, instead of property members.
     * @tparam T The target type for reflexive-introspection.
     * @return The total number of property members within the type.
     */
    template <typename T, size_t ...I>
    inline constexpr auto count(...) noexcept -> size_t
    {
        return sizeof...(I) - 1;
    }

    template <typename T, size_t ...I, typename = decltype(T{decoy<T, I>()...})>
    inline constexpr auto count(int) noexcept -> size_t
    {
        return reflection::count<T, I..., sizeof...(I)>(0);
    }
    /**#@-*/

    /**
     * Extracts the types of the property members of a type to be reflected upon.
     * @tparam T The target type for reflexive-introspection.
     * @return The tuple of extracted types.
     */
    template <typename T, size_t ...I, typename = decltype(T{decoy<T, I>()...})>
    inline constexpr auto loophole(std::index_sequence<I...>)
        -> detail::tuple<decltype(latch(tag<T, I>()))...>;

    /**
     * Retrieves a tuple with the internal member types of the target type.
     * @tparam T The target type for reflexive-introspection.
     * @return The tuple of extracted types.
     */
    template <typename T>
    inline constexpr auto loophole()
        -> decltype(loophole<T>(std::make_index_sequence<reflection::count<T>(0)>()));

    /**
     * Reflects over the target data type, that is it iterates over and extracts
     * information about the member properties of the target type.
     * @tparam T The target data type to be introspected.
     * @since 1.0
     */
    template <typename T>
    class reflector
    {
        static_assert(!std::is_union<T>::value, "union types cannot be reflected");
        static_assert(std::is_trivial<T>::value, "only trivial types can be reflected");

        private:
            /**
             * Transforms each tuple type into an aligned storage type.
             * @tparam U The tuple elements' type list.
             */
            template <typename ...U>
            inline static constexpr auto to_storage_tuple(detail::tuple<U...>)
                -> detail::tuple<typename std::aligned_storage<sizeof(U), alignof(U)>::type...>;

        public:
            using reflection_tuple = decltype(reflection::loophole<T>());
            using storage_tuple = decltype(to_storage_tuple(std::declval<reflection_tuple>()));

        static_assert(
            sizeof(reflection_tuple) == sizeof(T) && alignof(reflection_tuple) == alignof(T)
          , "the produced reflection tuple is not compatible with the target type");

        public:
            /**
             * Retrieves the number of members within the reflected type.
             * @return The number of members composing the target type.
             */
            inline static constexpr auto count() noexcept -> size_t
            {
                return reflection_tuple::count;
            }

            /**
             * Retrieves the memory offset of a member within the reflected type.
             * @tparam N The index of required member.
             * @return The target member's memory offset.
             */
            template <size_t N>
            inline static constexpr auto offset(const storage_tuple& t = {}) noexcept -> ptrdiff_t
            {
                return reinterpret_cast<size_t>(&t.template get<N>())
                     - reinterpret_cast<size_t>(&t.template get<0>());
            }
    };
}

MPIWCPP17_END_NAMESPACE

MPIWCPP17_DISABLE_GCC_WARNING_END("-Wnon-template-friend")

#endif
