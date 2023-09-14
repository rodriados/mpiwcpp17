/**
 * A thin C++17 wrapper for MPI.
 * @file MPI graph communicator wrapper.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <mpi.h>

#include <set>
#include <vector>
#include <cstdint>
#include <numeric>
#include <utility>
#include <algorithm>

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/process.hpp>

#include <mpiwcpp17/detail/topology.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace topology
{
    /**
     * A graph topology blueprint responsible for describing the connections between
     * the processes of a graph-topology communicator.
     * @see mpiwcpp17::topology::graph
     * @since 1.0
     */
    class graph_t : public detail::topology::blueprint_t
    {
        private:
            using node_t = process_t;
            using edge_t = std::pair<node_t, node_t>;

        private:
            std::set<edge_t> m_edges = {};

        private:
            template <typename T>
            using valid_iterator_t = std::void_t<
                decltype(std::declval<T&>().begin())
              , decltype(std::declval<T&>().end())>;

        public:
            inline graph_t() = default;
            inline graph_t(const graph_t&) = default;
            inline graph_t(graph_t&&) = default;

            /**
             * Initializes a new blueprint from a list of graph edges.
             * @param edges The list of edges to initialize the blueprint with.
             */
            inline graph_t(const std::initializer_list<edge_t>& edges)
              : m_edges (edges.begin(), edges.end())
            {}

            /**
             * Initializes a new blueprint from a container of graph edges.
             * @tparam C The container type to get graph edges from.
             * @param edges The container of edges to initialize the blueprint with.
             */
            template <typename C, typename = valid_iterator_t<C>>
            inline graph_t(const C& edges)
              : m_edges (edges.begin(), edges.end())
            {}

            inline graph_t& operator=(const graph_t&) = default;
            inline graph_t& operator=(graph_t&&) = default;

        public:
            inline raw_t commit(const raw_t&, bool = true) const override;
            inline static auto extract(const raw_t&) -> graph_t;

        public:
            /**
             * Returns an iterator pointing to the first edge of the topology.
             * @return An iterator to the beginning of topology's edges.
             */
            inline auto begin() const noexcept -> decltype(auto)
            {
                return m_edges.begin();
            }

            /**
             * Returns an iterator pointing after the last edge of the topology.
             * @return An iterator to the end of topology's edges.
             */
            inline auto end() const noexcept -> decltype(auto)
            {
                return m_edges.end();
            }

            /**
             * Inserts a new directed edge to the communicator graph blueprint.
             * @param x The origin process of the new edge.
             * @param y The destination process of the new edge.
             */
            inline void insert(node_t x, node_t y)
            {
                m_edges.insert(edge_t(x, y));
            }

            /**
             * Inserts new directed edges to the communicator graph blueprint.
             * @param edges The list of edges to be added to the graph blueprint.
             */
            inline void insert(const std::initializer_list<edge_t>& edges)
            {
                m_edges.insert(edges.begin(), edges.end());
            }

            /**
             * Inserts new directed edges to the communicator graph blueprint from
             * a container instance containing graph edges.
             * @tparam C The container type to insert new graph edges from.
             * @param edges The container of edges to be added to the graph blueprint.
             */
            template <typename C, typename = valid_iterator_t<C>>
            inline void insert(const C& edges)
            {
                m_edges.insert(edges.begin(), edges.end());
            }

            /**
             * Removes an existing edge from the communicator graph blueprint.
             * @param x The origin process of the edge to be removed.
             * @param y The destination process of the edge to be removed.
             */
            inline void remove(node_t x, node_t y)
            {
                m_edges.erase(edge_t(x, y));
            }

            /**
             * Removes existing edges from the communicator graph blueprint.
             * @param edges The edges to be removed from graph blueprint.
             */
            inline void remove(const std::initializer_list<edge_t>& edges)
            {
                for (const auto& edge : edges)
                    m_edges.erase(edge);
            }

            /**
             * Removes existing edges from the communicator graph blueprint.
             * @tparam C The container type of edges to remove from graph.
             * @param edges The container of edges to be removed from graph blueprint.
             */
            template <typename C, typename = valid_iterator_t<C>>
            inline void remove(const C& edges)
            {
                for (const auto& edge : edges)
                    m_edges.erase(edge);
            }
    };

    /**
     * Commits the described blueprint and creates a new communicator.
     * @param comm The original communicator to apply the blueprint to.
     * @param reorder May process ranks be reassigned within new communicator?
     * @return The new topology-applied communicator.
     */
    inline auto graph_t::commit(const raw_t& comm, bool reorder) const -> raw_t
    {
        int32_t it = 0, count = 0;

        for (auto [x, y] : m_edges)
            count = std::max({x, y, count});

        int32_t *index = new int32_t[++count]();
        int32_t *edges = new int32_t[m_edges.size()];

        for (auto [x, y] : m_edges)
            { ++index[x]; edges[it++] = y; }

        raw_t x;
        std::partial_sum(index, index + count, index);
        guard(MPI_Graph_create(comm, count, index, edges, reorder, &x));

        delete[] index;
        delete[] edges;
        return x;
    }

    /**
     * Retrieves the graph-topology blueprint applied over the communicator's processes.
     * @param comm The topology-enabled communicator to extract topology from.
     * @return The topology extracted from the given communicator.
     */
    inline auto graph_t::extract(const raw_t& comm) -> graph_t
    {
        int32_t node_count, edge_count;
        guard(MPI_Graphdims_get(comm, &node_count, &edge_count));

        int32_t *index = new int32_t[node_count]();
        int32_t *edges = new int32_t[edge_count]();
        guard(MPI_Graph_get(comm, node_count, edge_count, index, edges));

        auto edge_list = std::vector<edge_t>();

        for (int32_t i = 0, j = 0; i < node_count && j < edge_count; ++i)
            while (j < index[i]) edge_list.push_back({i, edges[j++]});

        delete[] index;
        delete[] edges;
        return graph_t(edge_list);
    }
}

MPIWCPP17_END_NAMESPACE
