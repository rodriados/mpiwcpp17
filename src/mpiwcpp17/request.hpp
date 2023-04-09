/**
 * A thin C++17 wrapper for MPI.
 * @file MPI asynchronous operation request payload register.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2023-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <tuple>
#include <utility>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/payload.hpp>
#include <mpiwcpp17/status.hpp>

MPIWCPP17_BEGIN_NAMESPACE

template <typename T = void>
class request_t;

template <>
class request_t<void>
{
    public:
        using raw_t = MPI_Request;
        using return_t = struct {};

    protected:
        raw_t m_request;

    public:
        inline request_t() = default;
        inline request_t(const request_t&) noexcept = default;
        inline request_t(request_t&&) noexcept = default;

        inline request_t& operator=(const request_t&) noexcept = default;
        inline request_t& operator=(request_t&&) noexcept = default;

        inline operator raw_t&() noexcept
        {
            return m_request;
        }

        inline operator raw_t*() noexcept
        {
            return &m_request;
        }

        inline explicit constexpr operator return_t() const noexcept
        {
            return {};
        }
};

template <typename T>
class request_t : public request_t<void>
{
    public:
        using return_t = typename payload_t<T>::return_t;

    protected:
        return_t m_payload;

    public:
        inline request_t() = default;
        inline request_t(const request_t&) noexcept = default;
        inline request_t(request_t&&) noexcept = default;

        inline request_t(const payload_t<T>& payload) noexcept
          : m_payload (payload)
        {}

        inline request_t(payload_t<T>&& payload) noexcept
          : m_payload (std::forward<decltype(payload)>(payload))
        {}

        inline request_t& operator=(const request_t&) noexcept = default;
        inline request_t& operator=(request_t&&) noexcept = default;

        inline operator return_t&() noexcept
        {
            return m_payload;
        }

        inline operator const return_t&() const noexcept
        {
            return m_payload;
        }
};

/*
 * Deduction guides for requests.
 * @since 1.0
 */
template <typename T> request_t(const payload_t<T>&) -> request_t<T>;
template <typename T> request_t(payload_t<T>&&) -> request_t<T>;

namespace request
{
    inline void cancel(request_t<>::raw_t& rq)
    {
        guard(MPI_Cancel(&rq));
    }

    template <typename T>
    inline auto status(request_t<T>& rq) -> std::tuple<bool, status_t>
    {
        int completed;
        status_t s; guard(MPI_Request_get_status(rq, &completed, s));
        return std::make_tuple(completed, s);
    }
}

MPIWCPP17_END_NAMESPACE
