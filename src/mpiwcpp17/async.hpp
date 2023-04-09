/**
 * A thin C++17 wrapper for MPI.
 * @file MPI asynchronous operation functions and utilities.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2023-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <tuple>
#include <utility>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/request.hpp>
#include <mpiwcpp17/status.hpp>
#include <mpiwcpp17/guard.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace async
{
    template <typename T>
    inline auto wait(request_t<T>& rq) -> typename request_t<T>::return_t
    {
        using return_t = typename request_t<T>::return_t;
        guard(MPI_Wait(rq, status::ignore));
        return static_cast<return_t>(rq);
    }

    template <typename ...T>
    inline auto waitall(request_t<T>&... rq) -> std::tuple<typename request_t<T>::return_t...>
    {
        typename request_t<>::raw_t rqs[] = {rq...};
        guard(MPI_Waitall(sizeof...(T), rqs, status::ignore));
        return std::make_tuple(static_cast<typename request_t<T>::return_t>(rq)...);
    }

    // template <typename ...T>
    // inline auto waitall(request_t<T>&... r)
    // -> std::tuple<typename request_t<T>::return_t...>
    // {
    //     typename request_t<>::raw_t rs[] = {r...};
    //     typename status_t::raw_t ss[] = {r...};
    //     guard(MPI_Waitall(sizeof...(T), rs, ss));
    //     return std::make_tuple(static_cast<typename request_t<T>::return_t>(r)...);
    // }
}

MPIWCPP17_END_NAMESPACE
