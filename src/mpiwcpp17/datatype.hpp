/**
 * A thin C++17 wrapper for MPI.
 * @file MPI datatype descriptors and describers.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <array>
#include <cstdint>
#include <utility>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/guard.hpp>

#include <mpiwcpp17/detail/tracker.hpp>
#include <mpiwcpp17/detail/attribute.hpp>
#include <mpiwcpp17/thirdparty/reflector.h>

MPIWCPP17_BEGIN_NAMESPACE
MPIWCPP17_FWD_GLOBAL_STATUS_FUNCTIONS

/**
 * The type for a datatype identifier instance. An instance of a datatype identifier
 * must exist for all types that are to trasit via MPI.
 * @since 3.0
 */
using datatype_t = MPI_Datatype;

namespace datatype
{
    /**
     * Declares datatype attribute namespace and corresponding functions.
     * @since 3.0
     */
    MPIWCPP17_ATTRIBUTE_DECLARE(
        datatype_t
      , MPI_Type_create_keyval, MPI_Type_free_keyval
      , MPI_Type_get_attr, MPI_Type_set_attr, MPI_Type_delete_attr
      , MPI_TYPE_DUP_FN, MPI_TYPE_NULL_DELETE_FN
    )

    /**
     * The datatype identity provider for a specified type. A custom, and possibly
     * generic, provider can be specified by specializing it to the target type.
     * @param T The type to be identified for MPI operations.
     * @since 3.0
     */
    template <typename T>
    struct provider_t;

    /**
     * Identifies the given type by retrieving its datatype identifier.
     * @tparam T The type to be identified.
     * @return The requested type's identifier.
     */
    template <typename T>
    MPIWCPP17_INLINE datatype_t identify()
    {
        static_assert(!std::is_union<T>::value, "union types cannot be used with MPI");
        static_assert(!std::is_reference<T>::value, "references cannot be used with MPI");

        static datatype_t id = detail::tracker_t::add(provider_t<T>::provide(), &MPI_Type_free);
        return id;
    }

    /**#@+
     * Specializations for identifiers of built-in types. These native types have
     * their identities built-in within MPI and can be used directly.
     * @since 1.0
     */
    template <> MPIWCPP17_INLINE datatype_t identify<bool>()        { return MPI_C_BOOL; }
    template <> MPIWCPP17_INLINE datatype_t identify<short>()       { return MPI_SHORT; }
    template <> MPIWCPP17_INLINE datatype_t identify<int>()         { return MPI_INT; }
    template <> MPIWCPP17_INLINE datatype_t identify<long>()        { return MPI_LONG; }
    template <> MPIWCPP17_INLINE datatype_t identify<float>()       { return MPI_FLOAT; }
    template <> MPIWCPP17_INLINE datatype_t identify<double>()      { return MPI_DOUBLE; }
    template <> MPIWCPP17_INLINE datatype_t identify<long long>()   { return MPI_LONG_LONG; }
    template <> MPIWCPP17_INLINE datatype_t identify<long double>() { return MPI_LONG_DOUBLE; }

    template <> MPIWCPP17_INLINE datatype_t identify<char>()          { return MPI_CHAR; }
    template <> MPIWCPP17_INLINE datatype_t identify<signed char>()   { return MPI_SIGNED_CHAR; }
    template <> MPIWCPP17_INLINE datatype_t identify<unsigned char>() { return MPI_UNSIGNED_CHAR; }
    template <> MPIWCPP17_INLINE datatype_t identify<wchar_t>()       { return MPI_WCHAR; }

    template <> MPIWCPP17_INLINE datatype_t identify<unsigned short>()     { return MPI_UNSIGNED_SHORT; }
    template <> MPIWCPP17_INLINE datatype_t identify<unsigned int>()       { return MPI_UNSIGNED; }
    template <> MPIWCPP17_INLINE datatype_t identify<unsigned long>()      { return MPI_UNSIGNED_LONG; }
    template <> MPIWCPP17_INLINE datatype_t identify<unsigned long long>() { return MPI_UNSIGNED_LONG_LONG; }
    /**#@-*/

    /**
     * Duplicates the identifier of a given datatype identifier instance.
     * @param type The datatype identifier to be duplicated.
     * @return A clone of the given datatype identifier.
     */
    MPIWCPP17_INLINE datatype_t duplicate(datatype_t type)
    {
        datatype_t dup; guard(MPI_Type_dup(type, &dup));
        return detail::tracker_t::add(dup, &MPI_Type_free);
    }

    /**
     * Informs the total size in bytes of a concrete type instance when represented
     * by its datatype identifier.
     * @param type The type's identifier.
     * @return The concrete type's size in bytes.
     */
    MPIWCPP17_INLINE size_t size(datatype_t type)
    {
        int size; guard(MPI_Type_size(type, &size));
        return static_cast<size_t>(size);
    }

    /**
     * Frees up the resources needed for storing types' identifiers. Effectively,
     * after destruction, these type identities are in an invalid state and must
     * not be used.
     * @param type The type identifier to be freed.
     */
    MPIWCPP17_INLINE void free(datatype_t type)
    {
        if (!finalized() && !detail::tracker_t::remove(type))
            guard(MPI_Type_free(&type));
    }

    namespace detail
    {
        /**
         * Describes a type from its internal member properties' types.
         * @tparam M The list of member properties' types.
         * @param offset The list of member properties' offsets.
         * @return The resulting type description identifier.
         */
        template <typename ...M>
        MPIWCPP17_INLINE datatype_t build_from_members(const std::array<ptrdiff_t, sizeof...(M)>& offset)
        {
            datatype_t result;
            constexpr const size_t count = sizeof...(M);

            // Describing a struct type by acquiring a type identity and the offset
            // of each of its member properties. If a property of an array type
            // has been found, than we also inform the array's element count.
            int blocks[count] = {std::max(std::extent_v<M>, 1ul)...};
            datatype_t members[count] = {identify<std::remove_extent_t<M>>()...};
            const MPI_Aint *offsets = (MPI_Aint*) offset.data();

            guard(MPI_Type_create_struct(count, blocks, offsets, members, &result));
            guard(MPI_Type_commit(&result));

            return result;
        }
    }

  #ifndef MPIWCPP17_AVOID_THIRDPARTY_REFLECTOR
    /**
     * Provides datatype description by using the automatic reflection mechanism.
     * @param T The type to be identified for MPI operations.
     * @since 3.0
     */
    template <typename T>
    struct provider_t
    {
        using reflection_t = ::REFLECTOR_NAMESPACE::reflection_t<T>;
        using reflection_tuple_t = typename reflection_t::reflection_tuple_t;

        /**
         * Provides a datatype description for the target type via reflection.
         * @return The target datatype descriptor identifier.
         */
        MPIWCPP17_INLINE static datatype_t provide()
        {
            return provide_by_reflection(std::make_index_sequence<reflection_t::count>());
        }

        /**
         * Provides the properties' description of a MPI-enabled datatype by using
         * member-wise reflection over the specified type.
         * @tparam I The indexes of properties within the type.
         * @return The target datatype descriptor identifier.
         */
        template <size_t ...I>
        MPIWCPP17_INLINE static datatype_t provide_by_reflection(std::index_sequence<I...>)
        {
            constexpr auto offset = std::array {reflection_t::template offset<I>()...};
            return detail::build_from_members<typename reflection_tuple_t::template element_t<I>...>(offset);
        }
    };
  #endif

    /**
     * Provides the properties' description of a MPI-enabled datatype.
     * @tparam T The datatype to be described.
     * @tparam R The properties' types of the target datatype.
     * @param member The target type member properties' pointers.
     * @return The target datatype identifier instance.
     */
    template <typename T, typename ...R>
    MPIWCPP17_INLINE datatype_t provide(R T::*... member)
    {
        const auto offset = std::array {(((char*)&(((T*)0x80)->*member)) - ((char*)0x80))...};
        return detail::build_from_members<R...>(offset);
    }
}

MPIWCPP17_END_NAMESPACE
