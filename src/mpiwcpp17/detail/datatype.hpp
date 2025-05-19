/**
 * A thin C++17 wrapper for MPI.
 * @file MPI datatype implementation detail.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2025-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <utility>
#include <cstddef>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/guard.hpp>

MPIWCPP17_BEGIN_NAMESPACE

/*
 * Auxiliary macro for declaring compile-time known mapping specializations for MPI
 * datatype identities of common scalar and built-in types.
 */
#define MPIWCPP17_TYPE_MAPPING template <> MPIWCPP17_INLINE MPI_Datatype

namespace detail::datatype
{
    /**
     * Auxiliary class for mapping between MPI datatype identities and payload types.
     * The mapper function must be specialized for each and every payload type that
     * will transit through one of MPI's native operations.
     * @since 2.1
     */
    struct mapper_t
    {
        /**
         * The generic datatype identity mapping getter.
         * @tparam T The payload type to be mapped to a MPI datatype identity.
         * @return The MPI datatype identity for the given payload type.
         */
        template <typename T> MPIWCPP17_INLINE static MPI_Datatype get();
    };

    /**#@+
     * Mapping specialization for common scalar types. These native types have their
     * datatype identities built-in within MPI and should be mapped statically.
     * @return The MPI datatype identity for the given payload type.
     * @since 2.1
     */
    MPIWCPP17_TYPE_MAPPING mapper_t::get<       bool       >() { return MPI_C_BOOL;             }
    MPIWCPP17_TYPE_MAPPING mapper_t::get<       char       >() { return MPI_CHAR;               }
    MPIWCPP17_TYPE_MAPPING mapper_t::get<    signed char   >() { return MPI_SIGNED_CHAR;        }
    MPIWCPP17_TYPE_MAPPING mapper_t::get<   unsigned char  >() { return MPI_UNSIGNED_CHAR;      }
    MPIWCPP17_TYPE_MAPPING mapper_t::get<      wchar_t     >() { return MPI_WCHAR;              }
    MPIWCPP17_TYPE_MAPPING mapper_t::get<        int       >() { return MPI_INT;                }
    MPIWCPP17_TYPE_MAPPING mapper_t::get<       long       >() { return MPI_LONG;               }
    MPIWCPP17_TYPE_MAPPING mapper_t::get<       short      >() { return MPI_SHORT;              }
    MPIWCPP17_TYPE_MAPPING mapper_t::get<       float      >() { return MPI_FLOAT;              }
    MPIWCPP17_TYPE_MAPPING mapper_t::get<      double      >() { return MPI_DOUBLE;             }
    MPIWCPP17_TYPE_MAPPING mapper_t::get<     long long    >() { return MPI_LONG_LONG;          }
    MPIWCPP17_TYPE_MAPPING mapper_t::get<    long double   >() { return MPI_LONG_DOUBLE;        }
    MPIWCPP17_TYPE_MAPPING mapper_t::get<   unsigned int   >() { return MPI_UNSIGNED;           }
    MPIWCPP17_TYPE_MAPPING mapper_t::get<   unsigned long  >() { return MPI_UNSIGNED_LONG;      }
    MPIWCPP17_TYPE_MAPPING mapper_t::get<  unsigned short  >() { return MPI_UNSIGNED_SHORT;     }
    MPIWCPP17_TYPE_MAPPING mapper_t::get<unsigned long long>() { return MPI_UNSIGNED_LONG_LONG; }
    /**#@-*/

    /**
     * Helper for describing the memory layout of a property member of a type to
     * be mapped for use with MPI operations. To map a type member, it is required
     * to provide the member type, its extent count and memory offset.
     * @since 2.1
     */
    using member_description_t = std::tuple<MPI_Datatype, size_t, ptrdiff_t>;

    /**
     * Builds a new MPI datatype identity from the raw memory layout description
     * of the member properties within a generic payload type.
     * @tparam N The number of member properties within the payload type.
     * @param layout The memory layout description of the type member properties.
     * @return The new datatype identity for the described payload type.
     */
    template <size_t N>
    MPIWCPP17_INLINE auto map_from_memory_layout(const member_description_t (&layout)[N])
    {
        int counts[N];
        MPI_Aint offsets[N];
        MPI_Datatype t, types[N];

        // A new type identity is described by acquiring a MPI type identity for
        // its members, the array length and the corresponding offset. Therefore,
        // these types are not constructed through their usual constructors by MPI,
        // but rather have their raw contents memory-copied into a new instance.
        for (size_t i = 0; i < N; ++i) {
            std::tie(types[i], counts[i], offsets[i]) = layout[i];
        }

        guard(MPI_Type_create_struct(N, counts, offsets, types, &t));
        guard(MPI_Type_commit(&t));

        return t;
    }
}

#undef MPIWCPP17_TYPE_MAPPING

MPIWCPP17_END_NAMESPACE
