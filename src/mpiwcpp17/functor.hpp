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

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/datatype.hpp>
#include <mpiwcpp17/guard.hpp>

#include <mpiwcpp17/detail/deferrer.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace functor
{
    /**
     * The type for a operator functor instance identifier. An identifier is needed
     * for a functor to be used as operator for a reduce collective operation.
     * @since 3.0
     */
    using raw_t = MPI_Op;

    /**
     * Creates the description of an operator functor, allowing it to be used with
     * MPI collective operations. The registered functor must comply with the same
     * type as requested by MPI's interfaces.
     * @see functor::create
     * @since 1.0
     */
    class descriptor_t
    {
        private:
            const raw_t m_funcid;

        private:
            MPIWCPP17_INLINE descriptor_t() noexcept = delete;
            MPIWCPP17_INLINE descriptor_t(const descriptor_t&) noexcept = delete;
            MPIWCPP17_INLINE descriptor_t(descriptor_t&&) noexcept = delete;

            /**
             * Constructs a new functor description and registers its identifier
             * into the static list of identities for future destruction.
             * @param funcid A functor's identity.
             */
            MPIWCPP17_INLINE descriptor_t(raw_t funcid) noexcept
              : m_funcid (funcid)
            {
                s_funcids.push_back(m_funcid);
            }

            MPIWCPP17_INLINE descriptor_t& operator=(const descriptor_t&) noexcept = delete;
            MPIWCPP17_INLINE descriptor_t& operator=(descriptor_t&&) noexcept = delete;

            MPIWCPP17_INLINE static void destroy();

        public:
            using lambda_t = void(void*, void*, int*, datatype::raw_t*);

        public:
            MPIWCPP17_INLINE static descriptor_t build(lambda_t*, bool = false);

            /**
             * Exposes the underlying MPI operator identifier, allowing a descriptor
             * to be used seamlessly with native MPI functions.
             * @return The internal MPI operator identifier.
             */
            MPIWCPP17_INLINE operator raw_t() const noexcept
            {
                return m_funcid;
            }

        private:
            MPIWCPP17_INLINE static std::vector<raw_t> s_funcids;
            MPIWCPP17_INLINE static auto _d = detail::deferrer_t(descriptor_t::destroy);
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
        void wrapper(void *a, void *b, int *count, datatype::raw_t*)
        {
            auto f = F();
            auto x = static_cast<T*>(a);
            auto y = static_cast<T*>(b);

            static_assert(std::is_assignable<T&, decltype((f)(*x, *y))>::value
              , "the given operator return type is different from expected");

            for (int i = 0; i < *count; ++i, ++x, ++y) {
                *y = (f)(*x, *y);
            }
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
    MPIWCPP17_INLINE raw_t create(bool commutative = false)
    {
        static descriptor_t description = descriptor_t::build(&detail::wrapper<T, F>, commutative);
        return (raw_t) description;
    }

    /**#@+
     * Registration of readily available MPI operator functors. These operators
     * can be directly used in operations with the types they are built to.
     * @since 1.0
     */
    MPIWCPP17_INLINE static constexpr const raw_t max     = MPI_MAX;
    MPIWCPP17_INLINE static constexpr const raw_t min     = MPI_MIN;
    MPIWCPP17_INLINE static constexpr const raw_t add     = MPI_SUM;
    MPIWCPP17_INLINE static constexpr const raw_t mul     = MPI_PROD;
    MPIWCPP17_INLINE static constexpr const raw_t andl    = MPI_LAND;
    MPIWCPP17_INLINE static constexpr const raw_t andb    = MPI_BAND;
    MPIWCPP17_INLINE static constexpr const raw_t orl     = MPI_LOR;
    MPIWCPP17_INLINE static constexpr const raw_t orb     = MPI_BOR;
    MPIWCPP17_INLINE static constexpr const raw_t xorl    = MPI_LXOR;
    MPIWCPP17_INLINE static constexpr const raw_t xorb    = MPI_BXOR;
    MPIWCPP17_INLINE static constexpr const raw_t minloc  = MPI_MINLOC;
    MPIWCPP17_INLINE static constexpr const raw_t maxloc  = MPI_MAXLOC;
    MPIWCPP17_INLINE static constexpr const raw_t replace = MPI_REPLACE;
    /**#@-*/

    /**
     * Builds a new functor operator descriptor.
     * @param f The functor to be registered for use with MPI collectives.
     * @param commutative Is the operator being registered commutative?
     */
    MPIWCPP17_INLINE descriptor_t descriptor_t::build(lambda_t *f, bool commutative)
    {
        raw_t funcid;
        guard(MPI_Op_create(f, commutative, &funcid));
        return descriptor_t(funcid);
    }

    /**
     * Frees up the resources needed for storing operator functors' descriptions.
     * Effectively, after destruction, these operators are in invalid state.
     * @see mpi::functor::descriptor_t
     */
    MPIWCPP17_INLINE void descriptor_t::destroy()
    {
        for (raw_t& funcid : s_funcids) {
            guard(MPI_Op_free(&funcid));
        }
    }
}

MPIWCPP17_END_NAMESPACE
