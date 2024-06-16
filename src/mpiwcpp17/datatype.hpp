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

#include <mpiwcpp17/detail/deferrer.hpp>
#include <mpiwcpp17/thirdparty/supertuple.hpp>
#include <mpiwcpp17/thirdparty/reflector.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace datatype
{
    /**
     * The type for a datatype identifier instance. An instance of a datatype descriptor
     * must exist for all types that are to trasit via MPI.
     * @since 3.0
     */
    using raw_t = MPI_Datatype;

    /**
     * Creates the description for a type that may transit within a MPI message.
     * @see datatype::describe
     * @since 1.0
     */
    class descriptor_t
    {
        private:
            const raw_t m_typeid;

        private:
            MPIWCPP17_INLINE descriptor_t() noexcept = delete;
            MPIWCPP17_INLINE descriptor_t(const descriptor_t&) noexcept = delete;
            MPIWCPP17_INLINE descriptor_t(descriptor_t&&) noexcept = delete;

            /**
             * Constructs a new type description and register the type identity
             * into the static list of identities for future destruction.
             * @param type A type's identity.
             */
            MPIWCPP17_INLINE descriptor_t(raw_t type) noexcept
              : m_typeid (type)
            {
                s_typeids.push_back(m_typeid);
            }

            MPIWCPP17_INLINE descriptor_t& operator=(const descriptor_t&) noexcept = delete;
            MPIWCPP17_INLINE descriptor_t& operator=(descriptor_t&&) noexcept = delete;

            MPIWCPP17_INLINE static void destroy();

        public:
            template <typename ...T>
            MPIWCPP17_INLINE static descriptor_t build(const std::array<ptrdiff_t, sizeof...(T)>&);

            /**
             * Exposes the underlying raw MPI datatype identifier, allowing a descriptor
             * to be used seamlessly with native MPI functions.
             * @return The internal MPI datatype identifier.
             */
            MPIWCPP17_INLINE operator raw_t() const noexcept
            {
                return m_typeid;
            }

        private:
            MPIWCPP17_INLINE static std::vector<raw_t> s_typeids;
            MPIWCPP17_INLINE static auto _d = detail::deferrer_t(descriptor_t::destroy);
    };

    /**
     * Describes a type and allows it to be sent to different processes via MPI.
     * @tparam T The type to be described.
     * @return The target type's descriptor instance.
     * @see datatype::descriptor_t
     */
    template <typename T>
    MPIWCPP17_INLINE descriptor_t describe();

    /**
     * Identifies the given type by retrieving its raw datatype identifier.
     * @tparam T The type to be identified.
     * @return The requested type's identifier.
     */
    template <typename T>
    MPIWCPP17_INLINE raw_t identify()
    {
        static_assert(!std::is_union<T>::value, "union types cannot be used with MPI");
        static_assert(!std::is_reference<T>::value, "references cannot be used with MPI");

        static descriptor_t description = describe<T>();
        return (raw_t) description;
    }

    /**#@+
     * Specializations for identifiers of built-in types. These native types have
     * their identities built-in within MPI and can be used directly.
     * @since 1.0
     */
    template <> MPIWCPP17_INLINE raw_t identify<bool>()     { return MPI_C_BOOL; }
    template <> MPIWCPP17_INLINE raw_t identify<char>()     { return MPI_CHAR; }
    template <> MPIWCPP17_INLINE raw_t identify<float>()    { return MPI_FLOAT; }
    template <> MPIWCPP17_INLINE raw_t identify<double>()   { return MPI_DOUBLE; }
    template <> MPIWCPP17_INLINE raw_t identify<int8_t>()   { return MPI_INT8_T; }
    template <> MPIWCPP17_INLINE raw_t identify<int16_t>()  { return MPI_INT16_T; }
    template <> MPIWCPP17_INLINE raw_t identify<int32_t>()  { return MPI_INT32_T; }
    template <> MPIWCPP17_INLINE raw_t identify<int64_t>()  { return MPI_INT64_T; }
    template <> MPIWCPP17_INLINE raw_t identify<uint8_t>()  { return MPI_UINT8_T; }
    template <> MPIWCPP17_INLINE raw_t identify<uint16_t>() { return MPI_UINT16_T; }
    template <> MPIWCPP17_INLINE raw_t identify<uint32_t>() { return MPI_UINT32_T; }
    template <> MPIWCPP17_INLINE raw_t identify<uint64_t>() { return MPI_UINT64_T; }
    template <> MPIWCPP17_INLINE raw_t identify<wchar_t>()  { return MPI_WCHAR; }
    /**#@-*/

    /**
     * Informs the total size in bytes of a concrete type instance when represented
     * by its datatype identifier.
     * @param type The type's identifier.
     * @return The concrete type's size in bytes.
     */
    MPIWCPP17_INLINE size_t size(raw_t type)
    {
        int result; guard(MPI_Type_size(type, &result));
        return static_cast<size_t>(result);
    }

    /**
     * Provides the properties' description of a MPI-enabled datatype.
     * @tparam T The datatype to be described.
     * @tparam R The properties' types of the target datatype.
     * @param members The target type member properties' pointers.
     * @return The target datatype descriptor instance.
     */
    template <typename T, typename ...R>
    MPIWCPP17_INLINE descriptor_t provide(R T::*... members)
    {
        return descriptor_t::build<R...>({
            ((char*) &(((T*) 0x80)->*members))
          - ((char*) 0x80)...
        });
    }

  #if !defined(MPIWCPP17_AVOID_THIRDPARTY_REFLECTOR)
    namespace detail {
        /**
         * Provides the properties' description of a MPI-enabled datatype by using
         * reflection over the specified type.
         * @tparam T The type to be provided as descriptor.
         * @tparam I The indexes of properties within the type.
         * @return The target datatype descriptor instance.
         */
        template <typename T, size_t ...I>
        MPIWCPP17_INLINE descriptor_t provide_by_reflection(std::index_sequence<I...>)
        {
            return descriptor_t::build<std::tuple_element_t<I,
                typename reflector::reflection_t<T>::reflection_tuple_t>...>(
                std::array {reflector::reflection_t<T>::template offset<I>()...}
            );
        }
    }

    /**
     * Creates a MPI type description for using reflection over the target type.
     * @tparam T The type to be described.
     * @return The target type's description instance.
     */
    template <typename T>
    MPIWCPP17_INLINE descriptor_t describe()
    {
        return detail::provide_by_reflection<T>(
            std::make_index_sequence<reflector::reflection_t<T>::count>()
        );
    }

  #else
MPIWCPP17_DISABLE_GCC_WARNING_BEGIN("-Wreturn-type")
    /**
     * Throws a compile-time error as no descriptor can be found for the given type.
     * In order to the compilation to be successful, either reflection must not
     * be avoided or this function must be manually specialized for the type.
     * @tparam T The type to be described.
     * @return The target type's descriptor instance.
     */
    template <typename T>
    MPIWCPP17_INLINE descriptor_t describe()
    {
        static_assert(
            !std::is_void<T>::value && std::is_void<T>::value
          , "no descriptor was found to the requested type so it cannot be used with MPI");
    }
MPIWCPP17_DISABLE_GCC_WARNING_END("-Wreturn-type")
  #endif

    /**
     * Describes a type from its internal member properties' types.
     * @tparam T The list of member properties' types.
     * @param array The list of member properties' offsets.
     * @return The resulting type description identifier.
     */
    template <typename ...T>
    MPIWCPP17_INLINE descriptor_t descriptor_t::build(const std::array<ptrdiff_t, sizeof...(T)>& array)
    {
        raw_t typedesc;
        constexpr const size_t count = sizeof...(T);

        // Describing a struct type by acquiring a type identity and the offset
        // of each of its member properties. If a property of an array type
        // has been found, than we also inform the array's element count.
        int blocks[count]  = {(std::extent_v<T> > 1 ? std::extent_v<T> : 1)...};
        raw_t types[count] = {identify<std::remove_extent_t<T>>()...};
        const MPI_Aint *offsets = array.data();

        guard(MPI_Type_create_struct(count, blocks, offsets, types, &typedesc));
        guard(MPI_Type_commit(&typedesc));

        return descriptor_t(typedesc);
    }

    /**
     * Frees up the resources needed for storing types' descriptions. Effectively,
     * after destruction, these type identities are in an invalid state and must
     * not be used.
     * @see mpi::datatype::descriptor_t
     */
    MPIWCPP17_INLINE void descriptor_t::destroy()
    {
        for (raw_t& typedesc : s_typeids) {
            guard(MPI_Type_free(&typedesc));
        }
    }
}

MPIWCPP17_END_NAMESPACE
