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

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/guard.hpp>

#include <mpiwcpp17/detail/tuple.hpp>
#include <mpiwcpp17/detail/reflection.hpp>

MPIWCPP17_BEGIN_NAMESPACE

/**
 * The type for a datatype identifier instance. An instance of a datatype descriptor
 * must exist for all types that are to trasit via MPI.
 * @since 1.0
 */
using datatype_t = MPI_Datatype;

namespace datatype
{
    /**
     * Creates the description for a type that may transit within a MPI message.
     * A descriptor must receive the type's member properties pointers as constructor
     * parameters in order to describe a type.
     * @see datatype::describe
     * @since 1.0
     */
    class descriptor_t
    {
        private:
            datatype_t m_typeid;

        private:
            inline static std::vector<datatype_t> s_typeids;

        public:
            inline descriptor_t() noexcept = delete;
            inline descriptor_t(const descriptor_t&) noexcept = delete;
            inline descriptor_t(descriptor_t&&) noexcept = delete;

            template <typename T, typename ...U>
            inline descriptor_t(U T::*...);

            template <size_t ...I, typename ...T>
            inline descriptor_t(const mpiwcpp17::detail::tuple_t<std::index_sequence<I...>, T...>&);

            inline descriptor_t& operator=(const descriptor_t&) noexcept = delete;
            inline descriptor_t& operator=(descriptor_t&&) noexcept = delete;

            inline operator datatype_t() const noexcept;
            inline static void destroy();

        private:
            inline descriptor_t(datatype_t) noexcept;
    };

MPIWCPP17_DISABLE_GCC_WARNING_BEGIN("-Wreturn-type")

    /**
     * Describes a type and allows it to be sent to different processes via MPI.
     * @tparam T The type to be described.
     * @return The target type's descriptor instance.
     * @see datatype::descriptor_t
     */
    template <typename T>
    inline descriptor_t describe();

  #if !defined(MPIWCPP17_AVOID_REFLECTION)
    /**
     * Creates a MPI type description for using reflection over the target type.
     * @tparam T The type to be described.
     * @return The target type's description instance.
     */
    template <typename T>
    inline descriptor_t describe()
    {
        return descriptor_t(typename detail::reflection::reflector_t<T>::reflection_tuple_t());
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
    inline descriptor_t describe()
    {
        static_assert(
            !std::is_void<T>::value && std::is_void<T>::value
          , "no descriptor was found to the requested type so it cannot be used with MPI"
        );
    }
  #endif

MPIWCPP17_DISABLE_GCC_WARNING_END("-Wreturn-type")

    /**
     * Identifies the given type by retrieving its raw datatype identifier.
     * @tparam T The type to be identified.
     * @return The requested type's identifier.
     */
    template <typename T>
    inline auto identify() -> datatype_t
    {
        static_assert(!std::is_union<T>::value, "union types cannot be used with MPI");
        static_assert(!std::is_reference<T>::value, "references cannot be used with MPI");

        static auto description = describe<T>();
        return (datatype_t) description;
    }

    /**
     * Informs the total size in bytes of a concrete type instance when represented
     * by its datatype identifier.
     * @param type The type's identifier.
     * @return The concrete type's size in bytes.
     */
    inline size_t size(datatype_t type)
    {
        int result; guard(MPI_Type_size(type, &result));
        return static_cast<size_t>(result);
    }

    /**#@+
     * Specializations for identifiers of built-in types. These native types have
     * their identities built-in within MPI and can be used directly.
     * @since 1.0
     */
    template <> inline datatype_t identify<bool>()     { return MPI_C_BOOL; };
    template <> inline datatype_t identify<char>()     { return MPI_CHAR; };
    template <> inline datatype_t identify<float>()    { return MPI_FLOAT; };
    template <> inline datatype_t identify<double>()   { return MPI_DOUBLE; };
    template <> inline datatype_t identify<int8_t>()   { return MPI_INT8_T; };
    template <> inline datatype_t identify<int16_t>()  { return MPI_INT16_T; };
    template <> inline datatype_t identify<int32_t>()  { return MPI_INT32_T; };
    template <> inline datatype_t identify<int64_t>()  { return MPI_INT64_T; };
    template <> inline datatype_t identify<uint8_t>()  { return MPI_UINT8_T; };
    template <> inline datatype_t identify<uint16_t>() { return MPI_UINT16_T; };
    template <> inline datatype_t identify<uint32_t>() { return MPI_UINT32_T; };
    template <> inline datatype_t identify<uint64_t>() { return MPI_UINT64_T; };
    template <> inline datatype_t identify<wchar_t>()  { return MPI_WCHAR; };
    /**#@-*/

    namespace detail
    {
        /**
         * Describes a type from its internal member properties' types.
         * @tparam T The list of member properties' types.
         * @param array The list of member properties' offsets.
         * @return The resulting type description identifier.
         */
        template <typename ...T>
        inline static auto describe(const std::array<ptrdiff_t, sizeof...(T)>& array) -> datatype_t
        {
            datatype_t result;
            constexpr const size_t count = sizeof...(T);

            // Describing a struct type by acquiring a type identity and the offset
            // of each of its member properties. If a property of an array type
            // has been found, than we also inform the array's element count.
            int blocks[count] = {(std::extent<T>::value > 1 ? std::extent<T>::value : 1)...};
            datatype_t types[count] = {identify<typename std::remove_extent<T>::type>()...};
            const MPI_Aint *offsets = array.data();

            guard(MPI_Type_create_struct(count, blocks, offsets, types, &result));
            guard(MPI_Type_commit(&result));

            return result;
        }
    }

    /**
     * Constructs a new type description. A type description is needed whenever
     * a message of non-builtin types must transit between MPI processes.
     * @tparam T The type to be described through its property members.
     * @tparam U The target member properties' types list.
     * @param members The target type member properties' pointers.
     */
    template <typename T, typename ...U>
    inline descriptor_t::descriptor_t(U T::*... members)
      : descriptor_t (detail::describe<U...>({
            ((char*) &(((T*) 0x80)->*members))
          - ((char*) 0x80) ...
        }))
    {}

    /**
     * Constructs a new type description from a tuple. The given tuple must contain
     * aligned member properties to those of the original type.
     * @tparam T The list of member properties' types within the tuple.
     * @param tuple A type-describing tuple instance.
     */
    template <size_t ...I, typename ...T>
    inline descriptor_t::descriptor_t(
        const mpiwcpp17::detail::tuple_t<std::index_sequence<I...>, T...>& t
    ) : descriptor_t (detail::describe<T...>({
            ((char*) &t.template get<I>())
          - ((char*) &t.template get<0>()) ...
        }))
    {}

    /**
     * Constructs a new type description and register the type identity into the
     * static list of identities for future destruction.
     * @param type A type's identity.
     */
    inline descriptor_t::descriptor_t(datatype_t type) noexcept
      : m_typeid (type)
    {
        s_typeids.push_back(type);
    }

    /**
     * Exposes the underlying raw MPI datatype identifier, allowing a descriptor
     * to be used seamlessly with native MPI functions.
     * @return The internal MPI datatype identifier.
     */
    inline descriptor_t::operator datatype_t() const noexcept
    {
        return m_typeid;
    }

    /**
     * Frees up the resources needed for storing types' descriptions. Effectively,
     * after destruction, these type identities are in an invalid state and must
     * not be used.
     * @see mpi::datatype::descriptor_t
     */
    inline void descriptor_t::destroy()
    {
        for (auto& type : s_typeids)
            guard(MPI_Type_free(&type));
    }
}

MPIWCPP17_END_NAMESPACE
