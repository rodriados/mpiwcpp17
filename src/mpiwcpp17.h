/**
 * A thin C++17 wrapper for MPI.
 * @file The project's main header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpiwcpp17/version.h>
#include <mpiwcpp17/environment.h>

#include <mpiwcpp17/global.hpp>
#include <mpiwcpp17/initiator.hpp>

#include <mpiwcpp17/error.hpp>
#include <mpiwcpp17/exception.hpp>
#include <mpiwcpp17/status.hpp>
#include <mpiwcpp17/datatype.hpp>
#include <mpiwcpp17/memory.hpp>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/collective.hpp>
#include <mpiwcpp17/functor.hpp>
