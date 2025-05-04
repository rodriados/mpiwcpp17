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

#include <mpiwcpp17/detail/handle.hpp>

MPIWCPP17_BEGIN_NAMESPACE

/**
 * Produces a function for attribute creation.
 * @tparam type The data type the attribute is associated to.
 * @tparam fcrt The function for attribute creation.
 * @tparam fdup The function for attribute duplication.
 * @tparam fdel The function for attribute destruction.
 */
#define MPIWCPP17_ATTRIBUTE_DECLARE_CREATE(type, fcrt, fdup, fdel)                          \
  MPIWCPP17_INLINE attribute_t create() {                                                   \
    int attr; guard(fcrt(fdup, fdel, &attr, nullptr));                                      \
    return attribute_t(attr, true);                                                         \
  }

/**
 * Produces a function for attribute value retrieval from data type.
 * @tparam type The data type the attribute is associated to.
 * @tparam fget The function for attribute value retrieval.
 * @param target The target object the attribute is attached to.
 * @param attr The attribute to retrieve from the given target object.
 * @return A flag of whether the attribute exists and the corresponding value.
 */
#define MPIWCPP17_ATTRIBUTE_DECLARE_GET(type, fget)                                         \
  template <typename T = void>                                                              \
  MPIWCPP17_INLINE std::pair<bool, T*> get(const type& target, const attribute_t& attr) {   \
    int g; T* ptr; guard(fget(target, attr, (void*) &ptr, &g));                             \
    return std::make_pair(g, ptr);                                                          \
  }

/**
 * Produces a function for attribute modification in data type.
 * @tparam type The data type the attribute is associated to.
 * @tparam fset The function for attribute modification.
 * @tparam T The type of the attribute value to be attached.
 * @param target The target object the attribute must be attached to.
 * @param value The attribute's value in relation to the given object.
 */
#define MPIWCPP17_ATTRIBUTE_DECLARE_SET(type, fset)                                         \
  template <typename T = void>                                                              \
  MPIWCPP17_INLINE void set(const type& target, const attribute_t& attr, T *value) {        \
    guard(fset(target, attr, (void*) value));                                               \
  }

/**
 * Produces a function for attribute removal from data type.
 * @tparam type The data type the attribute is associated to.
 * @tparam frem The function for attribute removal from data type.
 * @param target The target object the attribute must be disattached from.
 * @param attr The attribute to be disattached from object.
 */
#define MPIWCPP17_ATTRIBUTE_DECLARE_REMOVE(type, frem)                                      \
  MPIWCPP17_INLINE void remove(const type& target, const attribute_t& attr) {               \
    guard(frem(target, attr));                                                              \
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
  struct attribute_t : MPIWCPP17_INHERIT_HANDLE(int, ffre);                         \
  namespace attribute {                                                             \
    MPIWCPP17_ATTRIBUTE_DECLARE_CREATE(type, fcrt, fdup, fdel)                      \
    MPIWCPP17_ATTRIBUTE_DECLARE_REMOVE(type, frem)                                  \
    MPIWCPP17_ATTRIBUTE_DECLARE_GET(type, fget)                                     \
    MPIWCPP17_ATTRIBUTE_DECLARE_SET(type, fset)                                     \
  }

MPIWCPP17_END_NAMESPACE
