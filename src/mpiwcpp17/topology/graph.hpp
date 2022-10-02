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
#include <utility>

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
    class graph : public detail::topology::blueprint
    {
        private:
            using node_type = process::rank;
            using edge_type = std::pair<node_type, node_type>;

        private:
            std::set<edge_type> m_edges = {};

        private:
            template <typename T>
            using valid_iterator = std::void_t<
                decltype(std::declval<T&>().begin())
              , decltype(std::declval<T&>().end())>;

        public:
            inline graph() = default;
            inline graph(const graph&) = default;
            inline graph(graph&&) = default;

            /**
             * Initializes a new blueprint from a list of graph edges.
             * @param edges The list of edges to initialize the blueprint with.
             */
            inline graph(const std::initializer_list<edge_type>& edges)
              : m_edges (edges.begin(), edges.end())
            {}

            /**
             * Initializes a new blueprint from a container of graph edges.
             * @tparam C The container type to get graph edges from.
             * @param edges The container of edges to initialize the blueprint with.
             */
            template <typename C, typename = valid_iterator<C>>
            inline graph(const C& edges)
              : m_edges (edges.begin(), edges.end())
            {}

            inline graph& operator=(const graph&) = default;
            inline graph& operator=(graph&&) = default;

        public:
            inline raw_type commit(const raw_type&, bool = true) const override;
            inline static auto extract(const raw_type&) -> graph;

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
            inline void insert(node_type x, node_type y)
            {
                m_edges.insert(edge_type(x, y));
            }

            /**
             * Inserts new directed edges to the communicator graph blueprint.
             * @param edges The list of edges to be added to the graph blueprint.
             */
            inline void insert(const std::initializer_list<edge_type>& edges)
            {
                m_edges.insert(edges.begin(), edges.end());
            }

            /**
             * Inserts new directed edges to the communicator graph blueprint from
             * a container instance containing graph edges.
             * @tparam C The container type to insert new graph edges from.
             * @param edges The container of edges to be added to the graph blueprint.
             */
            template <typename C, typename = valid_iterator<C>>
            inline void insert(const C& edges)
            {
                m_edges.insert(edges.begin(), edges.end());
            }

            /**
             * Removes an existing edge from the communicator graph blueprint.
             * @param x The origin process of the edge to be removed.
             * @param y The destination process of the edge to be removed.
             */
            inline void remove(node_type x, node_type y)
            {
                m_edges.erase(edge_type(x, y));
            }

            /**
             * Removes existing edges from the communicator graph blueprint.
             * @param edges The edges to be removed from graph blueprint.
             */
            inline void remove(const std::initializer_list<edge_type>& edges)
            {
                for (const auto& edge : edges) { m_edges.erase(edge); }
            }

            /**
             * Removes existing edges from the communicator graph blueprint.
             * @tparam C The container type of edges to remove from graph.
             * @param edges The container of edges to be removed from graph blueprint.
             */
            template <typename C, typename = valid_iterator<C>>
            inline void remove(const C& edges)
            {
                for (const auto& edge : edges) { m_edges.erase(edge); }
            }
    };

    /**
     * Commits the described blueprint and creates a new communicator.
     * @param comm The original communicator to apply the blueprint to.
     * @param reorder May process ranks be reassigned within new communicator?
     * @return The new topology-applied communicator.
     */
    inline auto graph::commit(const raw_type& comm, bool reorder) const -> raw_type
    {
        int32_t it = 0, count = 0;

        for (auto [x, y] : m_edges)
            count = std::max({x, y, count});

        int32_t *index = new int32_t[++count]();
        int32_t *edges = new int32_t[m_edges.size()];

        for (auto [x, y] : m_edges)
            { ++index[x]; edges[it++] = y; }

        raw_type x;
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
    inline auto graph::extract(const raw_type& comm) -> graph
    {
        int32_t node_count, edge_count;
        guard(MPI_Graphdims_get(comm, &node_count, &edge_count));

        int32_t *index = new int32_t[node_count]();
        int32_t *edges = new int32_t[edge_count]();
        guard(MPI_Graph_get(comm, node_count, edge_count, index, edges));

        auto edge_list = std::vector<edge_type>();

        for (int32_t i = 0, j = 0; i < node_count && j < edge_count; ++i)
            while (j < index[i]) edge_list.push_back({i, edges[j++]});

        delete[] index;
        delete[] edges;
        return graph(edge_list);
    }
}

MPIWCPP17_END_NAMESPACE
