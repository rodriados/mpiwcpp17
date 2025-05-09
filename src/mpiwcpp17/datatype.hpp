/**
 * A thin C++17 wrapper for MPI.
 * @file MPI datatype descriptors and providers.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <array>
#include <cstddef>
#include <utility>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/guard.hpp>

#include <mpiwcpp17/detail/raii.hpp>
#include <mpiwcpp17/detail/handle.hpp>
#include <mpiwcpp17/detail/attribute.hpp>
#include <mpiwcpp17/detail/datatype.hpp>
#include <mpiwcpp17/thirdparty/reflector.h>

MPIWCPP17_BEGIN_NAMESPACE

/**
 * The raw MPI datatype identifier type.
 * This type is used to identify and describe a datatype, and its respective properties
 * that might transit through MPI collective operations.
 * @since 2.1
 */
struct datatype_t : MPIWCPP17_INHERIT_HANDLE(MPI_Datatype, MPI_Type_free);

/*
 * Auxiliary macros for implementing functions that wrap the creation of new datatypes.
 * The newly created datatypes are automatically attached to RAII.
 * @param x The datatype to be acquired by a handle.
 * @param B The call block to be wrapped.
 */
#define MPIWCPP17_TYPE_RAII(x)  datatype_t ((x), true)
#define MPIWCPP17_TYPE_CALL(B)  MPIWCPP17_TYPE_RAII(MPIWCPP17_GUARD_CALL(MPI_Datatype, B))

namespace datatype
{
    /**
     * Declares datatype attribute namespace and corresponding functions.
     * Attributes are identified by keys that can be used to attach and retrieve
     * generic data from datatypes.
     * @since 2.1
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
     * @since 2.1
     */
    template <typename T>
    struct provider_t;

    /**
     * Identifies the given type by retrieving its datatype identifier.
     * @tparam T The type to be identified.
     * @return The requested type's identifier.
     */
    template <typename T>
    MPIWCPP17_INLINE datatype_t::raw_t identify()
    {
        static_assert(!std::is_union_v<T>, "union types cannot be used with MPI");
        static_assert(!std::is_reference_v<T>, "references cannot be used with MPI");
        static_assert(!std::is_pointer_v<T>, "pointers cannot be used with MPI");
        return detail::datatype::mapper_t::get<std::remove_cv_t<T>>();
    }

    /**
     * Duplicates the identifier of a given datatype identifier instance.
     * @param type The datatype identifier to be duplicated.
     * @return A clone of the given datatype identifier.
     */
    MPIWCPP17_INLINE datatype_t duplicate(const datatype_t& type)
    {
        return MPIWCPP17_TYPE_CALL(MPI_Type_dup(type, &_));
    }

    /**
     * Informs the total size in bytes of a concrete type instance when represented
     * by its datatype identifier.
     * @param type The type's identifier.
     * @return The concrete type's size in bytes.
     */
    MPIWCPP17_INLINE size_t size(const datatype_t& type)
    {
        return (size_t) MPIWCPP17_GUARD_CALL(int, MPI_Type_size(type, &_));
    }

  #ifndef MPIWCPP17_AVOID_THIRDPARTY_REFLECTOR
    /**
     * Provides MPI datatype identities by using the automatic reflection mechanism.
     * @param T The payload type to be identified for MPI operations.
     * @since 2.1
     */
    template <typename T>
    struct provider_t
    {
        using reflection_t = ::REFLECTOR_NAMESPACE::reflection_t<T>;
        using reflection_tuple_t = typename reflection_t::reflection_tuple_t;

        /**
         * Provides a MPI datatype identity for the target payload type via reflection.
         * @return The MPI datatype identifier for the given payload type.
         */
        MPIWCPP17_INLINE static datatype_t provide()
        {
            return provide_by_reflection(std::make_index_sequence<reflection_t::count>());
        }

        /**
         * Describes the member properties to MPI-enable the target payload type
         * by using member-wise reflection to automatically build the new identity.
         * @tparam I The indexes of member properties within the type.
         * @return The MPI datatype identifier for the given payload type.
         */
        template <size_t ...I>
        MPIWCPP17_INLINE static datatype_t provide_by_reflection(std::index_sequence<I...>)
        {
            return detail::datatype::identify_from_memory_layout(std::array {
                std::make_tuple(
                    identify<std::tuple_element_t<I, reflection_tuple_t>>()
                  , /* reflection always normalizes array types */ 1UL
                  , reflection_t::template offset<I>())...
            });
        }
    };
  #endif

    /**
     * Provides the member properties to MPI-enable a payload type.
     * @tparam T The payload type to be described.
     * @tparam R The member property types of the given payload type.
     * @param member The payload type member property pointers.
     * @return The MPI datatype identifier for the given payload type.
     */
    template <typename T, typename ...R>
    MPIWCPP17_INLINE datatype_t provide(R T::*... member)
    {
        return detail::datatype::identify_from_memory_layout(std::array {
            std::make_tuple(
                identify<std::remove_extent_t<R>>()
              , std::max(std::extent_v<R>, 1UL)
              , reinterpret_cast<ptrdiff_t>(
                    std::addressof(reinterpret_cast<T*>(0)->*member)))...
        });
    }
}

/**
 * Generic mapping specialization for types with explicit memory layout providers.
 * If the payload type is not yet mapped to a MPI datatype identity, then a new datatype
 * identity is created from a known memory layout provider.
 * @tparam T The payload type to be mapped to a MPI datatype identity.
 * @return The MPI datatype identity for the given payload type.
 * @since 2.1
 */
template <typename T>
MPIWCPP17_INLINE datatype_t::raw_t detail::datatype::mapper_t::get()
{
    static datatype_t t = detail::raii_t::register_handle(
        mpiwcpp17::datatype::provider_t<T>::provide());
    return t;
}

#undef MPIWCPP17_TYPE_CALL
#undef MPIWCPP17_TYPE_RAII

MPIWCPP17_END_NAMESPACE
