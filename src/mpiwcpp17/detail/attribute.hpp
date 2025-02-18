/**
 * A thin C++17 wrapper for MPI.
 * @file Generic attribute functions generators.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <utility>

#include <mpiwcpp17/environment.h>
#include <mpiwcpp17/global.hpp>
#include <mpiwcpp17/guard.hpp>

#include <mpiwcpp17/detail/tracker.hpp>

MPIWCPP17_BEGIN_NAMESPACE

/**
 * Produces a function for attribute creation.
 * @tparam type The data type the attribute is associated to.
 * @tparam fcrt The function for attribute creation.
 * @tparam fdup The function for attribute duplication.
 * @tparam fdel The function for attribute destruction.
 * @tparam ffre The function for attribute resource release.
 */
#define MPIWCPP17_ATTRIBUTE_DECLARE_CREATE(type, fcrt, fdup, fdel, ffre)            \
  MPIWCPP17_INLINE attribute_t create() {                                           \
    attribute_t attr; guard(fcrt(fdup, fdel, &attr, nullptr));                      \
    detail::tracker_t::add(attr, &ffre);                                            \
    return attr;                                                                    \
  }

/**
 * Produces a function for attribute value retrieval from data type.
 * @tparam type The data type the attribute is associated to.
 * @tparam fget The function for attribute value retrieval.
 * @param target The target object the attribute is attached to.
 * @param attr The attribute to retrieve from the given target object.
 * @return A flag of whether the attribute exists and the corresponding value.
 */
#define MPIWCPP17_ATTRIBUTE_DECLARE_GET(type, fget)                                 \
  template <typename T = void>                                                      \
  MPIWCPP17_INLINE std::pair<bool, T*> get(type target, attribute_t attr) {         \
    int f; T* ptr; guard(fget(target, attr, (void*) &ptr, &f));                     \
    return std::make_pair(f, ptr);                                                  \
  }

/**
 * Produces a function for attribute modification in data type.
 * @tparam type The data type the attribute is associated to.
 * @tparam fset The function for attribute modification.
 * @tparam T The type of the attribute value to be attached.
 * @param target The target object the attribute must be attached to.
 * @param value The attribute's value in relation to the given object.
 */
#define MPIWCPP17_ATTRIBUTE_DECLARE_SET(type, fset)                                 \
  template <typename T = void>                                                      \
  MPIWCPP17_INLINE void set(type target, attribute_t attr, T *value) {              \
    guard(fset(target, attr, (void*) value));                                       \
  }

/**
 * Produces a function for attribute removal from data type.
 * @tparam type The data type the attribute is associated to.
 * @tparam frem The function for attribute removal from data type.
 * @param target The target object the attribute must be disattached from.
 * @param attr The attribute to be disattached from object.
 */
#define MPIWCPP17_ATTRIBUTE_DECLARE_REMOVE(type, frem)                              \
  MPIWCPP17_INLINE void remove(type target, attribute_t attr) {                     \
    guard(frem(target, attr));                                                      \
  }

/**
 * Produces a function for attribute's resources release.
 * @tparam type The data type the attribute is associated to.
 * @tparam ffre The function for attribute resource release.
 * @param attr The attribute to be destroyed and have its resources released.
 */
#define MPIWCPP17_ATTRIBUTE_DECLARE_FREE(type, ffree)                               \
  MPIWCPP17_INLINE void free(attribute_t attr) {                                    \
    if(!finalized() && !detail::tracker_t::remove(attr))                            \
      guard(ffree(&attr));                                                          \
  }

/**
 * Produces a namespace with a complete set of attribute functions.
 * @tparam type The data type the attribute is associated to.
 * @tparam fcrt The function for attribute creation.
 * @tparam ffre The function for freeing attribute resources.
 * @tparam fget The function for attribute value retrieval.
 * @tparam fset The function for attribute modification.
 * @tparam frem The function for attribute removal from data type.
 * @tparam fdup The function for attribute duplication.
 * @tparam fdel The function for attribute destruction.
 */
#define MPIWCPP17_ATTRIBUTE_DECLARE(type, fcrt, ffre, fget, fset, frem, fdup, fdel) \
  using attribute_t = int;                                                          \
  namespace attribute {                                                             \
    MPIWCPP17_ATTRIBUTE_DECLARE_CREATE(type, fcrt, fdup, fdel, ffre)                \
    MPIWCPP17_ATTRIBUTE_DECLARE_REMOVE(type, frem)                                  \
    MPIWCPP17_ATTRIBUTE_DECLARE_GET(type, fget)                                     \
    MPIWCPP17_ATTRIBUTE_DECLARE_SET(type, fset)                                     \
    MPIWCPP17_ATTRIBUTE_DECLARE_FREE(type, ffre)                                    \
  }

MPIWCPP17_END_NAMESPACE
