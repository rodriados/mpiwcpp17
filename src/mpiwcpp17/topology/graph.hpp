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

#include <mpiwcpp17/environment.hpp>
#include <mpiwcpp17/communicator.hpp>
#include <mpiwcpp17/process.hpp>
#include <mpiwcpp17/world.hpp>

MPIWCPP17_BEGIN_NAMESPACE

namespace topology
{
    /**
     * A graph topology communicator. This communicator allows the use of neighbor
     * operations, so that data is only sent to and from neighboring processes.
     * @since 1.0
     */
    class graph : public communicator
    {
        protected:
            using node_type = process::rank;
            using edge_type = std::pair<node_type, node_type>;

        public:
            class blueprint;

        public:
            inline constexpr graph() noexcept = default;
            inline graph(const graph&) = default;
            inline graph(graph&&) noexcept = default;

            inline graph(const raw_type&, const blueprint&, bool = true);

            inline graph& operator=(const graph&) = default;
            inline graph& operator=(graph&&) = default;

            inline blueprint topology() const;
    };

    /**
     * A graph topology blueprint responsible for describing the connections between
     * the processes of a graph-topology communicator.
     * @see mpiwcpp17::topology::graph
     * @since 1.0
     */
    class graph::blueprint
    {
        private:
            int32_t m_count = 0;
            std::vector<std::set<node_type>> m_adjacency = {};

        public:
            inline blueprint() = default;
            inline blueprint(const blueprint&) = default;
            inline blueprint(blueprint&&) = default;

            inline blueprint(int32_t);
            inline blueprint(int32_t, std::initializer_list<edge_type>);

            inline blueprint& operator=(const blueprint&) = default;
            inline blueprint& operator=(blueprint&&) = default;

            inline void add(node_type, node_type);
            inline void remove(node_type, node_type);

            inline raw_type commit(const raw_type&, bool = true) const;
    };

    /**
     * Initializes a new graph-topology communicator from a previously created communicator
     * and a topology blueprint to apply over the communicator.
     * @param comm The previously created and base communicator.
     * @param topology The graph blueprint to apply over the communicator.
     * @param reorder May the processes' ranks be reordered in the new communicator?
     */
    inline graph::graph(const raw_type& comm, const blueprint& topology, bool reorder)
      : communicator (topology.commit(comm, reorder))
    {}

    /**
     * Initializes a new blueprint from a graph composed of disconnected nodes.
     * @param count The number of nodes within graph.
     */
    inline graph::blueprint::blueprint(int32_t count)
      : m_count (count)
      , m_adjacency (count)
    {}

    /**
     * Initializes a new blueprint from a list of graph edges.
     * @param count The number of nodes within graph.
     * @param edges The list of edges to initialize the blueprint with.
     */
    inline graph::blueprint::blueprint(int32_t count, std::initializer_list<edge_type> edges)
      : blueprint (count)
    {
        for (const auto& edge : edges)
            add(edge.first, edge.second);
    }

    /**
     * Adds a new directed edge to the communicator graph blueprint.
     * @param x The origin process of the new edge.
     * @param y The destination process of the new edge.
     */
    inline void graph::blueprint::add(node_type x, node_type y)
    {
        if (x < m_count && y < m_count)
            m_adjacency[x].insert(y);
    }

    /**
     * Removes an existing edge from the communicator graph blueprint.
     * @param x The origin process of the edge to be removed.
     * @param y The destination process of the edge to be removed.
     */
    inline void graph::blueprint::remove(node_type x, node_type y)
    {
        if (x < m_count && y < m_count)
            m_adjacency[x].erase(y);
    }

    /**
     * Retrieves the graph-topology blueprint applied over the communicator's processes.
     * @return The blueprint describing the communicator's graph-topology.
     */
    inline auto graph::topology() const -> blueprint
    {
        struct { int32_t index, edges; } count;
        guard(MPI_Graphdims_get(this->m_comm, &count.index, &count.edges));

        blueprint topology (count.index);
        std::vector<int32_t> index (count.index), edges (count.edges);
        guard(MPI_Graph_get(this->m_comm, count.index, count.edges, index.data(), edges.data()));

        for (int x = 0, n = 0; x < count.index; ++x)
            while (n < index[x] && n < count.edges)
                topology.add(x, edges[n++]);

        return topology;
    }

    /**
     * Commits the described blueprint and creates a new communicator.
     * @param comm The original communicator to apply the blueprint to.
     * @param reorder May process ranks be reassigned within new communicator?
     * @return The new topology-applied communicator.
     */
    inline auto graph::blueprint::commit(const raw_type& comm, bool reorder) const -> raw_type
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
