/**
 * A thin C++17 wrapper for MPI.
 * @file MPI datatype descriptors and describers.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <array>
#include <vector>
#include <cstdint>
#include <utility>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/guard.hpp>

#include <mpiwcpp17/thirdparty/reflector.hpp>

MPIWCPP17_BEGIN_NAMESPACE

/**
 * The type for a datatype identifier instance. An instance of a datatype identifier
 * must exist for all types that are to trasit via MPI.
 * @since 3.0
 */
using datatype_t = MPI_Datatype;

namespace datatype
{
    /**
     * Describes a type and allows it to be sent to different processes via MPI.
     * @tparam T The type to be described.
     * @return The target type's identifier instance.
     */
    template <typename T>
    MPIWCPP17_INLINE datatype_t describe();

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

        static datatype_t identifier = describe<T>();
        return identifier;
    }

    /**#@+
     * Specializations for identifiers of built-in types. These native types have
     * their identities built-in within MPI and can be used directly.
     * @since 1.0
     */
    template <> MPIWCPP17_INLINE datatype_t identify<bool>()     { return MPI_C_BOOL; }
    template <> MPIWCPP17_INLINE datatype_t identify<char>()     { return MPI_CHAR; }
    template <> MPIWCPP17_INLINE datatype_t identify<float>()    { return MPI_FLOAT; }
    template <> MPIWCPP17_INLINE datatype_t identify<double>()   { return MPI_DOUBLE; }
    template <> MPIWCPP17_INLINE datatype_t identify<int8_t>()   { return MPI_INT8_T; }
    template <> MPIWCPP17_INLINE datatype_t identify<int16_t>()  { return MPI_INT16_T; }
    template <> MPIWCPP17_INLINE datatype_t identify<int32_t>()  { return MPI_INT32_T; }
    template <> MPIWCPP17_INLINE datatype_t identify<int64_t>()  { return MPI_INT64_T; }
    template <> MPIWCPP17_INLINE datatype_t identify<uint8_t>()  { return MPI_UINT8_T; }
    template <> MPIWCPP17_INLINE datatype_t identify<uint16_t>() { return MPI_UINT16_T; }
    template <> MPIWCPP17_INLINE datatype_t identify<uint32_t>() { return MPI_UINT32_T; }
    template <> MPIWCPP17_INLINE datatype_t identify<uint64_t>() { return MPI_UINT64_T; }
    template <> MPIWCPP17_INLINE datatype_t identify<wchar_t>()  { return MPI_WCHAR; }
    /**#@-*/

    /**
     * Duplicates the identifier of a given datatype identifier instance.
     * @param type The datatype identifier to be duplicated.
     * @return A clone of the given datatype identifier.
     */
    MPIWCPP17_INLINE datatype_t duplicate(const datatype_t& type)
    {
        datatype_t nt; guard(MPI_Type_dup(type, &nt));
        return nt;
    }

    /**
     * Informs the total size in bytes of a concrete type instance when represented
     * by its datatype identifier.
     * @param type The type's identifier.
     * @return The concrete type's size in bytes.
     */
    MPIWCPP17_INLINE size_t size(const datatype_t& type)
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
    MPIWCPP17_INLINE void free(datatype_t& type)
    {
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
            int blocks[count] = {std::max(std::extent_v<M>, 1)...};
            datatype_t members[count] = {identify<std::remove_extent_t<M>>()...};
            const MPI_Aint *offsets = (MPI_Aint*) offset.data();

            guard(MPI_Type_create_struct(count, blocks, offsets, types, &result));
            guard(MPI_Type_commit(&result));

            return result;
        }

      #if !defined(MPIWCPP17_AVOID_THIRDPARTY_REFLECTOR)
        /**
         * Provides the properties' description of a MPI-enabled datatype by using
         * reflection over the specified type.
         * @tparam T The type to be provided as descriptor.
         * @tparam I The indexes of properties within the type.
         * @return The target datatype descriptor instance.
         */
        template <typename T, size_t ...I>
        MPIWCPP17_INLINE datatype_t provide_by_reflection(std::index_sequence<I...>)
        {
            using reflection_t = reflector::reflection_t<T>;
            using elements_t = typename reflection_t::reflection_tuple_t;
            return build_from_members<std::tuple_element_t<I, elements_t>...>({
                reflection_t::template offset<I>()...
            });
        }
      #endif
    }

    /**
     * Provides the properties' description of a MPI-enabled datatype.
     * @tparam T The datatype to be described.
     * @tparam R The properties' types of the target datatype.
     * @param members The target type member properties' pointers.
     * @return The target datatype identifier instance.
     */
    template <typename T, typename ...R>
    MPIWCPP17_INLINE datatype_t provide(R T::*... members)
    {
        return detail::build_from_members<R...>({
            ((char*) &(((T*) 0x80)->*members))
          - ((char*) 0x80)...
        });
    }

MPIWCPP17_DISABLE_GCC_WARNING_BEGIN("-Wreturn-type")
  #if !defined(MPIWCPP17_AVOID_THIRDPARTY_REFLECTOR)
    /**
     * Creates a MPI type description for using reflection over the target type.
     * @tparam T The type to be described.
     * @return The target type's description instance.
     */
    template <typename T>
    MPIWCPP17_INLINE datatype_t describe()
    {
        return detail::provide_by_reflection<T>(
            std::make_index_sequence<reflector::reflection_t<T>::count>()
        );
    }
  #else
    /**
     * Throws a compile-time error as no descriptor can be found for the given type.
     * In order to the compilation to be successful, either reflection must not
     * be avoided or this function must be manually specialized for the type.
     * @tparam T The type to be described.
     * @return The target type's descriptor instance.
     */
    template <typename T>
    MPIWCPP17_INLINE datatype_t describe()
    {
        static_assert(
            !std::is_void<T>::value && std::is_void<T>::value
          , "no descriptor was found to the requested type so it cannot be used with MPI");
    }
  #endif
MPIWCPP17_DISABLE_GCC_WARNING_END("-Wreturn-type")
}

MPIWCPP17_END_NAMESPACE
