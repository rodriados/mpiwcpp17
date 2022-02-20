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

namespace datatype
{
    /**
     * The type for a datatype identifier instance. An instance of a datatype descriptor
     * must exist for all types that are to trasit via MPI.
     * @since 1.0
     */
    using id = MPI_Datatype;

    /**
     * Creates the description for a type that may transit within a MPI message.
     * A descriptor must receive the type's member properties pointers as constructor
     * parameters in order to describe a type.
     * @see datatype::describe
     * @since 1.0
     */
    class descriptor
    {
        private:
            datatype::id m_typeid;

        private:
            inline static std::vector<datatype::id> typeids;

        public:
            inline descriptor() noexcept = delete;
            inline descriptor(const descriptor&) noexcept = delete;
            inline descriptor(descriptor&&) noexcept = delete;

            template <typename T, typename ...U>
            inline descriptor(U T::*...);

            template <size_t ...I, typename ...T>
            inline descriptor(const mpiw::detail::tuple<std::index_sequence<I...>, T...>&);

            inline descriptor& operator=(const descriptor&) noexcept = delete;
            inline descriptor& operator=(descriptor&&) noexcept = delete;

            inline operator datatype::id() const noexcept;
            inline static void destroy();

        private:
            inline descriptor(datatype::id) noexcept;
    };

    /**
     * Describes a type and allows it to be sent to different processes via MPI.
     * @tparam T The type to be described.
     * @return The target type's descriptor instance.
     * @see datatype::descriptor
     */
    template <typename T>
    inline descriptor describe();

  #if !defined(MPIWCPP17_AVOID_REFLECTION)
    /**
     * Creates a MPI type description for using reflection over the target type.
     * @tparam T The type to be described.
     * @return The target type's description instance.
     */
    template <typename T>
    inline descriptor describe()
    {
        return descriptor(typename detail::reflection::reflector<T>::reflection_tuple());
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
    inline descriptor describe()
    {
        static_assert(
            !std::is_void<T>::value && std::is_void<T>::value
          , "no descriptor was found to the requested type so it cannot be used with MPI");
    }
  #endif

    /**
     * Identifies the given type by retrieving its raw datatype identifier.
     * @tparam T The type to be identified.
     * @return The requested type's identifier.
     */
    template <typename T>
    inline auto identify() -> datatype::id
    {
        static_assert(!std::is_union<T>::value, "union types cannot be used with MPI");
        static_assert(!std::is_reference<T>::value, "references cannot be used with MPI");
        static_assert(std::is_trivial<T>::value, "only trivial types can be used with MPI");

        static auto description = describe<T>();
        return (datatype::id) description;
    }

    /**
     * Informs the total size in bytes of a concrete type instance when represented
     * by its datatype identifier.
     * @param type The type's identifier.
     * @return The concrete type's size in bytes.
     */
    inline size_t size(datatype::id type)
    {
        int result;
        guard(MPI_Type_size(type, &result));
        return static_cast<size_t>(result);
    }

    /**#@+
     * Specializations for identifiers of built-in types. These native types have
     * their identities built-in within MPI and can be used directly.
     * @since 1.0
     */
    template <> inline datatype::id identify<bool>()     { return MPI_C_BOOL; };
    template <> inline datatype::id identify<char>()     { return MPI_CHAR; };
    template <> inline datatype::id identify<float>()    { return MPI_FLOAT; };
    template <> inline datatype::id identify<double>()   { return MPI_DOUBLE; };
    template <> inline datatype::id identify<int8_t>()   { return MPI_INT8_T; };
    template <> inline datatype::id identify<int16_t>()  { return MPI_INT16_T; };
    template <> inline datatype::id identify<int32_t>()  { return MPI_INT32_T; };
    template <> inline datatype::id identify<int64_t>()  { return MPI_INT64_T; };
    template <> inline datatype::id identify<uint8_t>()  { return MPI_UINT8_T; };
    template <> inline datatype::id identify<uint16_t>() { return MPI_UINT16_T; };
    template <> inline datatype::id identify<uint32_t>() { return MPI_UINT32_T; };
    template <> inline datatype::id identify<uint64_t>() { return MPI_UINT64_T; };
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
        inline static auto describe(const std::array<ptrdiff_t, sizeof...(T)>& array) -> datatype::id
        {
            datatype::id result;
            constexpr const size_t count = sizeof...(T);

            // Describing a struct type by acquiring a type identity and the offset
            // of each of its member properties. If a property of an array type
            // has been found, than we also inform the array's element count.
            int blocks[count] = {(std::extent<T>::value > 1 ? std::extent<T>::value : 1)...};
            datatype::id types[count] = {identify<typename std::remove_extent<T>::type>()...};
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
    inline descriptor::descriptor(U T::*... members)
      : descriptor (detail::describe<U...>({(char*) &(((T*)64)->*members) - ((char*)64)...}))
    {}

    /**
     * Constructs a new type description from a tuple. The given tuple must contain
     * aligned member properties to those of the original type.
     * @tparam T The list of member properties' types within the tuple.
     * @param tuple A type-describing tuple instance.
     */
    template <size_t ...I, typename ...T>
    inline descriptor::descriptor(const mpiw::detail::tuple<std::index_sequence<I...>, T...>& t)
      : descriptor (detail::describe<T...>({(char*) &t.template get<I>() - (char*) &t.template get<0>()...}))
    {}

    /**
     * Constructs a new type description and register the type identity into the
     * static list of identities for future destruction.
     * @param type A type's identity.
     */
    inline descriptor::descriptor(datatype::id type) noexcept
      : m_typeid (type)
    {
        typeids.push_back(type);
    }

    inline descriptor::operator datatype::id() const noexcept
    {
        return m_typeid;
    }

    /**
     * Frees up the resources needed for storing types' descriptions. Effectively,
     * after destruction, these type identities are in an invalid state and must
     * not be used.
     * @see mpi::datatype::descriptor
     */
    inline void descriptor::destroy()
    {
        for (auto& type : typeids) guard(MPI_Type_free(&type));
    }
}

MPIWCPP17_END_NAMESPACE
