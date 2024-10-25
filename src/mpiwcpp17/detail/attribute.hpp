/**
 * A thin C++17 wrapper for MPI.
 * @file Generic attribute functions generators.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <utility>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/guard.hpp>

#include <mpiwcpp17/detail/tracker.hpp>

MPIWCPP17_BEGIN_NAMESPACE
MPIWCPP17_FWD_GLOBAL_STATUS_FUNCTIONS

namespace detail
{
    /**
     * The generic attribute type. An attribute is simply an integer identifier
     * that specifies a key to a value in a data type.
     * @since 3.0
     */
    typedef int attribute_t;
}

/**
 * Produces a function for attribute creation.
 * @param type The data type the attribute is associated to.
 * @param fcreate The function for attribute creation.
 * @param fdup The function for attribute duplication.
 * @param fdel The function for attribute destruction.
 */
#define MPIWCPP17_ATTRIBUTE_DECLARE_CREATE(type, fcreate, fdup, fdel)               \
    MPIWCPP17_INLINE attribute_t create() {                                         \
      attribute_t attr; mpiwcpp17::guard(fcreate(fdup, fdel, &attr, nullptr));      \
      mpiwcpp17::detail::tracker_t::add(                                            \
          static_cast<uintptr_t>(attr)                                              \
        , &detail::f_delete_attr);                                                  \
      return attr;                                                                  \
    }

/**
 * Produces a function for attribute value retrieval from data type.
 * @param type The data type the attribute is associated to.
 * @param fget The function for attribute value retrieval.
 */
#define MPIWCPP17_ATTRIBUTE_DECLARE_GET(type, fget)                                 \
    template <typename T = void>                                                    \
    MPIWCPP17_INLINE std::pair<bool, T*> get(type target, attribute_t attr) {       \
      int flag; T* ptr; mpiwcpp17::guard(fget(target, attr, (void*) &ptr, &flag));  \
      return std::make_pair(flag, ptr);                                             \
    }

/**
 * Produces a function for attribute modification in data type.
 * @param type The data type the attribute is associated to.
 * @param fset The function for attribute modification.
 */
#define MPIWCPP17_ATTRIBUTE_DECLARE_SET(type, fset)                                 \
    template <typename T = void>                                                    \
    MPIWCPP17_INLINE void set(type target, attribute_t attr, T *value) {            \
      mpiwcpp17::guard(fset(target, attr, (void*) value));                          \
    }

/**
 * Produces a function for attribute removal from data type.
 * @param type The data type the attribute is associated to.
 * @param fremove The function for attribute removal from data type.
 */
#define MPIWCPP17_ATTRIBUTE_DECLARE_REMOVE(type, fremove)                           \
    MPIWCPP17_INLINE void remove(type target, attribute_t attr) {                   \
      mpiwcpp17::guard(fremove(target, attr));                                      \
    }

/**
 * Produces a function for attribute's resources release.
 * @param type The data type the attribute is associated to.
 * @param ffree The function for attribute resource release.
 */
#define MPIWCPP17_ATTRIBUTE_DECLARE_FREE(type, ffree)                               \
    MPIWCPP17_INLINE void free(attribute_t attr) {                                  \
      uintptr_t xattr = static_cast<uintptr_t>(attr);                               \
      if(!mpiwcpp17::finalized() && !mpiwcpp17::detail::tracker_t::remove(xattr))   \
        mpiwcpp17::guard(ffree(&attr));                                             \
    }

/**
 * Produces a namespace with a complete set of attribute functions.
 * @param type The data type the attribute is associated to.
 * @param init The function for attribute creation.
 * @param free The function for freeing attribute resources.
 * @param get The function for attribute value retrieval.
 * @param set The function for attribute modification.
 * @param rem The function for attribute removal from data type.
 * @param dup The function for attribute duplication.
 * @param del The function for attribute destruction.
 */
#define MPIWCPP17_ATTRIBUTE_DECLARE(type, init, free, get, set, rem, dup, del)      \
  using attribute_t = mpiwcpp17::detail::attribute_t;                               \
  namespace attribute {                                                             \
    namespace detail {                                                              \
      MPIWCPP17_INLINE int f_delete_attr(uintptr_t *xattr) {                        \
        attribute_t attr = static_cast<attribute_t>(*xattr);                        \
        return (free)(&attr);                                                       \
      }                                                                             \
    }                                                                               \
    MPIWCPP17_ATTRIBUTE_DECLARE_CREATE(type, init, dup, del)                        \
    MPIWCPP17_ATTRIBUTE_DECLARE_REMOVE(type, rem)                                   \
    MPIWCPP17_ATTRIBUTE_DECLARE_GET(type, get)                                      \
    MPIWCPP17_ATTRIBUTE_DECLARE_SET(type, set)                                      \
    MPIWCPP17_ATTRIBUTE_DECLARE_FREE(type, free)                                    \
  }

MPIWCPP17_END_NAMESPACE
