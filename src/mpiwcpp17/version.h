/**
 * A thin C++17 wrapper for MPI.
 * @file Compiler-time macros encoding mpiwcpp17 release version.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

/**
 * The preprocessor macros encoding the current mpiwcpp17 library release version.
 * This is guaranteed to change with every mpiwcpp17 release.
 */
#define MPIWCPP17_VERSION 20100

/**
 * The preprocessor macros encoding the release policy's values to the current mpiwcpp17
 * library release version.
 */
#define MPIWCPP17_VERSION_MAJOR (MPIWCPP17_VERSION / 10000)
#define MPIWCPP17_VERSION_MINOR (MPIWCPP17_VERSION / 100 % 100)
#define MPIWCPP17_VERSION_PATCH (MPIWCPP17_VERSION % 100)
/**#@-*/
