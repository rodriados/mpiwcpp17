/**
 * A thin C++17 wrapper for MPI.
 * @file A wrapper for MPI collective operator functors.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <utility>
#include <vector>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/datatype.hpp>
#include <mpiwcpp17/guard.hpp>

MPIWCPP17_BEGIN_NAMESPACE

/**
 * The type for a operator functor instance identifier. An identifier is needed
 * for a functor to be used as operator for a reduce collective operation.
 * @since 1.0
 */
using functor_t = MPI_Op;

namespace functor
{
    /**
     * The raw MPI operator functor interface.
     * @since 1.0
     */
    using raw_t = void(void*, void*, int*, datatype_t*);

    /**
     * Registers a operator functor, allowing to be used with MPI collective operations.
     * The registry functor must be the same type as requested by MPI's interfaces.
     * @see functor::create
     * @since 1.0
     */
    class registry_t
    {
        private:
            functor_t m_fid;

        private:
            inline static std::vector<functor_t> s_functors;

        public:
            inline registry_t() noexcept = delete;
            inline registry_t(const registry_t&) noexcept = delete;
            inline registry_t(registry_t&&) noexcept = delete;

            inline registry_t(raw_t*, bool = true);

            inline registry_t& operator=(const registry_t&) noexcept = delete;
            inline registry_t& operator=(registry_t&&) noexcept = delete;

            inline operator functor_t() const noexcept;
            inline static void destroy();
    };

    namespace detail
    {
        /**
         * Wraps a typed operator function by transforming it into a generic and
         * MPI-compatible operator.
         * @tparam T The type of data the operator works with.
         * @tparam F The functor type to be injected into and executed by the operator.
         * @param a The operation's first operand reference.
         * @param b The operation's second operand and output values reference.
         * @param count The total number of elements to process during execution.
         */
        template <typename T, typename F>
        void wrapper(void *a, void *b, int *count, datatype_t*)
        {
            auto f = F();
            auto x = static_cast<T*>(a);
            auto y = static_cast<T*>(b);

            static_assert(std::is_assignable<T&, decltype((f)(*x, *y))>::value
              , "the given operator return type is different from expected");

            for (int i = 0; i < *count; ++i, ++x, ++y)
                *y = (f)(*x, *y);
        }
    }

    /**
     * Registers a new operator functor within MPI's internal machinery, allowing
     * it to be used with collective operations.
     * @tparam T The operator's target operands' type.
     * @tparam F The functor type to be injected into and executed by the operator.
     * @param commutative Is the operator being created commutative?
     * @return The identifier of the created operator.
     */
    template <typename T, typename F>
    inline auto create(bool commutative = true) -> functor_t
    {
        static auto registration = registry_t(&detail::wrapper<T, F>, commutative);
        return (functor_t) registration;
    }

    /**#@+
     * Registration of readily available MPI operator functors. These operators
     * can be directly used in operations with the types they are built to.
     * @since 1.0
     */
    inline static constexpr const functor_t max     = MPI_MAX;
    inline static constexpr const functor_t min     = MPI_MIN;
    inline static constexpr const functor_t add     = MPI_SUM;
    inline static constexpr const functor_t mul     = MPI_PROD;
    inline static constexpr const functor_t andl    = MPI_LAND;
    inline static constexpr const functor_t andb    = MPI_BAND;
    inline static constexpr const functor_t orl     = MPI_LOR;
    inline static constexpr const functor_t orb     = MPI_BOR;
    inline static constexpr const functor_t xorl    = MPI_LXOR;
    inline static constexpr const functor_t xorb    = MPI_BXOR;
    inline static constexpr const functor_t minloc  = MPI_MINLOC;
    inline static constexpr const functor_t maxloc  = MPI_MAXLOC;
    inline static constexpr const functor_t replace = MPI_REPLACE;
    /**#@-*/

    /**
     * Registers a new operator functor.
     * @param f The functor to be registered for use with MPI collectives.
     * @param commutative Is the operator being registered commutative?
     */
    inline registry_t::registry_t(raw_t *f, bool commutative)
    {
        guard(MPI_Op_create(f, commutative, &m_fid));
        s_functors.push_back(m_fid);
    }

    /**
     * Exposes the underlying raw MPI operator identifier, allowing the registry
     * to be used seamlessly with native MPI functions.
     * @return The internal MPI operator identifier.
     */
    inline registry_t::operator functor_t() const noexcept
    {
        return m_fid;
    }

    /**
     * Frees up the resources needed for storing operator functors' descriptions.
     * Effectively, after destruction, these operators are in invalid state.
     * @see mpi::functor::registry_t
     */
    inline void registry_t::destroy()
    {
        for (auto& fid : s_functors)
            guard(MPI_Op_free(&fid));
    }
}

MPIWCPP17_END_NAMESPACE
