/**
 * A thin C++17 wrapper for MPI.
 * @file The project API declaration header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#ifndef MPIWCPP17_HEADER_INCLUDED
#define MPIWCPP17_HEADER_INCLUDED

#include <mpiwcpp17/version.h>
#include <mpiwcpp17/environment.h>

#include <mpiwcpp17/global.hpp>
#include <mpiwcpp17/engine.hpp>

#include <mpiwcpp17/error.hpp>
#include <mpiwcpp17/exception.hpp>
#include <mpiwcpp17/status.hpp>
#include <mpiwcpp17/info.hpp>
#include <mpiwcpp17/datatype.hpp>
#include <mpiwcpp17/policy.hpp>
#include <mpiwcpp17/functor.hpp>
#include <mpiwcpp17/window.hpp>
#include <mpiwcpp17/memory.hpp>
#include <mpiwcpp17/communicator.hpp>

#include <mpiwcpp17/operation/barrier.hpp>
#include <mpiwcpp17/operation/probe.hpp>
#include <mpiwcpp17/operation/send.hpp>
#include <mpiwcpp17/operation/receive.hpp>
#include <mpiwcpp17/operation/broadcast.hpp>
#include <mpiwcpp17/operation/allreduce.hpp>
#include <mpiwcpp17/operation/allgather.hpp>
#include <mpiwcpp17/operation/reduce.hpp>
#include <mpiwcpp17/operation/gather.hpp>
#include <mpiwcpp17/operation/scatter.hpp>

#endif
