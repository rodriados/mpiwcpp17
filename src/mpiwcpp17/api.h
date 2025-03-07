/**
 * A thin C++17 wrapper for MPI.
 * @file The project's API exposition header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#ifndef MPIWCPP17_HEADER_INCLUDED
#define MPIWCPP17_HEADER_INCLUDED

#include <mpiwcpp17/version.h>
#include <mpiwcpp17/environment.h>

#include <mpiwcpp17/global.hpp>
#include <mpiwcpp17/initiator.hpp>

#include <mpiwcpp17/error.hpp>
#include <mpiwcpp17/exception.hpp>
#include <mpiwcpp17/info.hpp>
#include <mpiwcpp17/status.hpp>
#include <mpiwcpp17/datatype.hpp>
#include <mpiwcpp17/flag.hpp>
#include <mpiwcpp17/functor.hpp>
#include <mpiwcpp17/memory.hpp>
#include <mpiwcpp17/communicator.hpp>

#include <mpiwcpp17/collective/barrier.hpp>
#include <mpiwcpp17/collective/broadcast.hpp>
#include <mpiwcpp17/collective/probe.hpp>
#include <mpiwcpp17/collective/send.hpp>
#include <mpiwcpp17/collective/receive.hpp>
#include <mpiwcpp17/collective/allreduce.hpp>
#include <mpiwcpp17/collective/reduce.hpp>
#include <mpiwcpp17/collective/allgather.hpp>
#include <mpiwcpp17/collective/gather.hpp>
#include <mpiwcpp17/collective/scatter.hpp>

#endif
