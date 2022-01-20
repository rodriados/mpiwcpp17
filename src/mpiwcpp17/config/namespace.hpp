/**
 * A thin C++17 wrapper for MPI.
 * @file Namespace configuration and macro definitions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

/**
 * Defines the namespace in which the library lives. This might be overriden if
 * the default namespace value is already in use.
 * @since 1.0
 */
#if defined(MPIWCPP17_OVERRIDE_NAMESPACE)
  #define MPIWCPP17_NAMESPACE MPIWCPP17_OVERRIDE_NAMESPACE
#else
  #define MPIWCPP17_NAMESPACE mpi
#endif

/**
 * This macro is used to open the MPIwCPP17 namespace block and may be overriden
 * if the user so wish or the `mpi::` namespace for some reason already exists.
 * @since 1.0
 */
#define MPIWCPP17_BEGIN_NAMESPACE               \
    namespace MPIWCPP17_NAMESPACE {             \
        inline namespace v1 {

/**
 * This macro is used to close the MPIwCPP17 namespace block and must not be in
 * any way overriden.
 * @since 1.0
 */
#define MPIWCPP17_END_NAMESPACE                 \
    }}
