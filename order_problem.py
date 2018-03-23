import numpy as np

from copy import copy


class OrderProblem:
    def __init__(self, dimension):
        self.dimension = dimension
        self.costs = -1 * np.ones(
            (self.dimension, self.dimension), dtype=np.int
        )


class OrderSolution:
    def __init__(self, problem):
        self.problem = problem
        self.dimension = self.problem.dimension
        self.edges_added = np.zeros(
            (self.dimension, self.dimension),
            dtype=np.bool
        )

        self.node_degrees = np.zeros(self.dimension, dtype=np.int)
        # Track which connected component each node belongs to.
        # Components are named after an arbitrary node in the component.
        self.connected_component = -1 * np.ones(self.dimension, dtype=np.int)
        self.feasible_edges = np.zeros(
            (self.dimension, self.dimension),
            dtype=np.bool
        )
        for i in range(self.dimension):
            self.feasible_edges[i, i+1:] = True
        # Used for error checking.
        self.valid_edges = copy(self.feasible_edges)

        self.ensure_validity()
        self.complete = False

    def ensure_validity(self):
        assert np.sum(self.edges_added * self.valid_edges, axis=(0, 1)) == \
            np.sum(self.edges_added, axis=(0, 1))

    def ensure_completion(self):
        self.ensure_validity()
        assert not self.feasible_edges.any()
        cc_name = self.connected_component[0]
        assert all(self.connected_component == cc_name)
        assert self.complete

    def add_edge(self, node_a, node_b):
        node_a, node_b = sorted([node_a, node_b])

        assert self.feasible_edges[node_a, node_b]
        assert not self.edges_added[node_a, node_b]
        self.feasible_edges[node_a, node_b] = False
        self.edges_added[node_a, node_b] = True

        for node in [node_a, node_b]:
            assert self.node_degrees[node] < 2
            self.node_degrees[node] += 1
            if self.node_degrees[node] == 2:
                # No more edges allowed for this node!
                self.feasible_edges[node, :] = False
                self.feasible_edges[:, node] = False

        # This check needs to happen before updating edge feasibility.
        # The TSPSolution class interprets
        # not self.complete and not self.feasible_edges.any()
        # to mean that we need just one final edge for a cycle.
        if not self.feasible_edges.any():
            self.complete = True
            return

        if self.connected_component[node_a] == -1 and \
                self.connected_component[node_b] == -1:
            self.connected_component[[node_a, node_b]] = node_a
        elif self.connected_component[node_a] != -1 and \
                self.connected_component[node_b] != -1:
            assert self.connected_component[node_a] != \
                self.connected_component[node_b]
            swallower, swallowed = [
                self.connected_component[node_a],
                self.connected_component[node_b]
            ]
            self.connected_component[
                [self.connected_component == swallowed]
            ] = swallower
            mask = self.connected_component == swallower
            # For every node in the new, combined component...
            for node in np.arange(self.dimension)[mask]:
                # Edges to other nodes in the component are now infeasible.
                self.feasible_edges[node, mask] = False
        else:
            if self.connected_component[node_a] == -1:
                addition = node_a
                swallower = self.connected_component[node_b]
            else:
                assert self.connected_component[node_b] == -1
                addition = node_b
                swallower = self.connected_component[node_a]
            assert swallower != -1
            self.connected_component[addition] = swallower
            mask = self.connected_component == swallower
            self.feasible_edges[addition, mask] = False
            self.feasible_edges[mask, addition] = False

        self.ensure_validity()

    def endpoints(self):
        return np.arange(self.dimension)[self.node_degrees == 1]

    def components(self):
        endpoints = self.endpoints()
        assert len(endpoints) % 2 == 0
        if len(endpoints) == 0:  # Either a complete loop, or no components
            endpoints = [0]  # Arbitrarily start at 0th node
        explored = set()
        result = []
        for endpoint in sorted(endpoints):
            if self.connected_component[endpoint] not in explored:
                explored.add(self.connected_component[endpoint])
                itinerary = []
                unvisited_neighbors = [endpoint]
                while unvisited_neighbors:
                    here = unvisited_neighbors[0]
                    itinerary.append(here)
                    neighbors = np.arange(self.dimension)[
                        (self.edges_added[here, :] | self.edges_added[:, here])
                    ]
                    unvisited_neighbors = [
                        n for n in neighbors if n not in itinerary
                    ]
                    # If a solution is valid, the only way for there to be two
                    # unvisited neighbors is if we just started on a loop.
                    assert len(unvisited_neighbors) <= 1 or \
                        (len(endpoints) == 1 and len(itinerary) == 1)
                # Close loop if it was a complete tour.
                if len(endpoints) == 1 and len(explored) > 1:
                    itinerary.append(itinerary[0])
                result.append(itinerary)
        return result
