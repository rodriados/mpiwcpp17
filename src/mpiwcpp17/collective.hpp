/**
 * A thin C++17 wrapper for MPI.
 * @file Helper header for including all MPI collective operations.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpiwcpp17/collective/barrier.hpp>
#include <mpiwcpp17/collective/broadcast.hpp>
#include <mpiwcpp17/collective/probe.hpp>
#include <mpiwcpp17/collective/send.hpp>
#include <mpiwcpp17/collective/receive.hpp>
#include <mpiwcpp17/collective/allreduce.hpp>
#include <mpiwcpp17/collective/reduce.hpp>
#include <mpiwcpp17/collective/allgather.hpp>
