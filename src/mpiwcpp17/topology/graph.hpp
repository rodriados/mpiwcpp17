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
            using underlying_type = detail::topology::blueprint;

        private:
            std::vector<std::set<node_type>> m_adjacency = {};

        public:
            inline graph() = default;
            inline graph(const graph&) = default;
            inline graph(graph&&) = default;

            inline graph(int32_t);
            inline graph(int32_t, const std::initializer_list<edge_type>&);

            inline graph& operator=(const graph&) = default;
            inline graph& operator=(graph&&) = default;

            inline void add(node_type, node_type);
            inline void remove(node_type, node_type);

            inline void extract(const raw_type&) override;
            inline raw_type commit(const raw_type&, bool = true) const override;
    };

    /**
     * Initializes a new blueprint from a graph composed of disconnected nodes.
     * @param count The number of nodes within graph.
     */
    inline graph::graph(int32_t count)
      : underlying_type (count)
      , m_adjacency (count)
    {}

    /**
     * Initializes a new blueprint from a list of graph edges.
     * @param count The number of nodes within graph.
     * @param edges The list of edges to initialize the blueprint with.
     */
    inline graph::graph(int32_t count, const std::initializer_list<edge_type>& edges)
      : graph (count)
    {
        for (const auto& edge : edges)
            add(edge.first, edge.second);
    }

    /**
     * Adds a new directed edge to the communicator graph blueprint.
     * @param x The origin process of the new edge.
     * @param y The destination process of the new edge.
     */
    inline void graph::add(node_type x, node_type y)
    {
        if (x < m_count && y < m_count)
            m_adjacency[x].insert(y);
    }

    /**
     * Removes an existing edge from the communicator graph blueprint.
     * @param x The origin process of the edge to be removed.
     * @param y The destination process of the edge to be removed.
     */
    inline void graph::remove(node_type x, node_type y)
    {
        if (x < m_count && y < m_count)
            m_adjacency[x].erase(y);
    }

    /**
     * Retrieves the graph-topology blueprint applied over the communicator's processes.
     * @param comm The topology-enabled communicator to extract topology from.
     */
    inline void graph::extract(const raw_type& comm)
    {
        int32_t edge_count;
        guard(MPI_Graphdims_get(comm, &m_count, &edge_count));

        int32_t *index = new int32_t[m_count];
        int32_t *edges = new int32_t[edge_count];
        m_adjacency = std::vector<std::set<node_type>>(m_count);

        guard(MPI_Graph_get(comm, m_count, edge_count, index, edges));

        for (int x = 0, previous = 0; x < m_count; previous = index[x++])
            m_adjacency[x].insert(edges + previous, edges + index[x]);

        delete[] index;
        delete[] edges;
    }

    /**
     * Commits the described blueprint and creates a new communicator.
     * @param comm The original communicator to apply the blueprint to.
     * @param reorder May process ranks be reassigned within new communicator?
     * @return The new topology-applied communicator.
     */
    inline auto graph::commit(const raw_type& comm, bool reorder) const -> raw_type
    {
        raw_type x;
        std::vector<node_type> edges, index;

        for (int32_t i = 0, total = 0; i < m_count; ++i) {
            const auto& neighbors = m_adjacency[i];
            edges.insert(edges.end(), neighbors.begin(), neighbors.end());
            index.push_back(total += neighbors.size());
        }

        guard(MPI_Graph_create(comm, m_count, index.data(), edges.data(), reorder, &x));
        return x;
    }
}

MPIWCPP17_END_NAMESPACE
