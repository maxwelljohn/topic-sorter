import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from collections import defaultdict

FIGSIZE = 6


class TSPProblem(object):
    def __init__(self, filepath):
        assert filepath.endswith('.tsp')
        with open(filepath, 'r') as infile:
            self.info = defaultdict(list)
            line = infile.readline()
            while not line.startswith('NODE_COORD_SECTION'):
                parts = line.split(':')
                self.info[parts[0].strip()].append(':'.join(parts[1:]).strip())
                line = infile.readline()
            assert self.info['TYPE'] == ['TSP']
            assert self.info['EDGE_WEIGHT_TYPE'] == ['EUC_2D']
            assert len(self.info['DIMENSION']) == 1
            self.dimension = int(self.info['DIMENSION'][0])

            self.nodes = np.zeros((self.dimension, 2))
            for i in range(1, self.dimension+1):
                line = infile.readline().rstrip()
                # Unfortunately the .tsp files don't distinguish between
                # north vs south latitude or east vs west longitude.
                # So the TSP may be flipped in some cases.
                node_num, latitude, longitude = line.split(' ')
                assert int(node_num) == i
                self.nodes[i-1] = np.array([float(longitude), float(latitude)])
            line = infile.readline().rstrip()
            assert line == '' or line == 'EOF'

        self.costs = np.zeros((self.dimension, self.dimension), dtype=np.int)
        for i in range(self.dimension):
            for j in range(i+1, self.dimension):
                self.costs[i, j] = round(
                    np.linalg.norm(self.nodes[i]-self.nodes[j])
                )

    def show(self, tour=None):
        point_xs = np.array([node[0] for node in self.nodes])
        point_ys = np.array([node[1] for node in self.nodes])
        plt.figure(figsize=(FIGSIZE, FIGSIZE))
        plt.plot(
            point_xs, point_ys, 'ro', markersize=3,
            scalex=True, scaley=True
        )

        if tour:
            endpoints = np.arange(tour.dimension)[tour.vertex_degrees == 1]
            assert len(endpoints) % 2 == 0
            if len(endpoints) == 0:  # Visualizing a complete tour
                endpoints = [0]  # Arbitrarily start at 0th node
            for endpoint in endpoints:  # Incomplete tours get drawn twice
                itinerary = []
                unvisited_neighbors = [endpoint]
                while unvisited_neighbors:
                    here = unvisited_neighbors[0]
                    itinerary.append(here)
                    neighbors = np.arange(tour.dimension)[
                        (tour.segments[here, :] | tour.segments[:, here]) == 1
                    ]
                    unvisited_neighbors = [
                        n for n in neighbors if n not in itinerary
                    ]
                    assert len(unvisited_neighbors) <= 1 or \
                        (len(endpoints) == 1 and len(itinerary) == 1)
                if len(endpoints) == 1:  # Close loop if it was a complete tour
                    itinerary.append(itinerary[0])
                itinerary_xs = point_xs[itinerary]
                itinerary_ys = point_ys[itinerary]
                plt.plot(
                    itinerary_xs, itinerary_ys, 'kx-', markersize=1
                )

        plt.show()


class TSPSolution(object):
    def __init__(self, problem, filepath=None):
        self.problem = problem
        self.dimension = self.problem.dimension
        self.segments = np.zeros(
            (self.dimension, self.dimension),
            dtype=np.int
        )

        self.vertex_degrees = np.zeros(self.dimension, dtype=np.int)
        # Track which connected component each vertex belongs to.
        # Components are named after an arbitrary vertex in the component.
        self.connected_component = -1 * np.ones(self.dimension, dtype=np.int)
        self.feasible_edges = np.zeros(
            (self.dimension, self.dimension),
            dtype=np.bool
        )
        for i in range(self.dimension):
            self.feasible_edges[i, i+1:] = True

        if filepath:
            assert filepath.endswith('.tour')
            with open(filepath, 'r') as infile:
                self.info = defaultdict(list)
                line = infile.readline()
                while not line.startswith('TOUR_SECTION'):
                    parts = line.split(':')
                    self.info[parts[0].strip()].append(
                        ':'.join(parts[1:]).strip()
                    )
                    line = infile.readline()
                assert self.info['TYPE'] == ['TOUR']
                assert len(self.info['DIMENSION']) == 1
                assert int(self.info['DIMENSION'][0]) == self.dimension

                line = infile.readline().rstrip()
                # .tour files are 1-indexed, but we 0-index nodes.
                first_node = int(line)-1
                latest_node = first_node
                for i in range(self.dimension-1):
                    line = infile.readline().rstrip()
                    this_node = int(line)-1
                    self.add_edge(latest_node, this_node)
                    latest_node = this_node
                line = infile.readline().rstrip()
                if line == '-1':
                    self.add_edge(latest_node, first_node)
                    line = infile.readline().rstrip()
                assert line == '' or line == 'EOF'

    def add_edge(self, node_a, node_b):
        node_a, node_b = sorted([node_a, node_b])

        assert self.feasible_edges[node_a, node_b]
        assert self.segments[node_a, node_b] == 0
        self.feasible_edges[node_a, node_b] = False
        self.segments[node_a, node_b] = 1

        for node in [node_a, node_b]:
            assert self.vertex_degrees[node] < 2
            self.vertex_degrees[node] += 1
            if self.vertex_degrees[node] == 2:
                # No more edges allowed for this node!
                self.feasible_edges[node, :] = False
                self.feasible_edges[:, node] = False

        if not self.feasible_edges.any():
            return  # The tour is complete.

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

        # If the tour is almost complete, make the loop-closing edge feasible.
        if not self.feasible_edges.any():
            endpoints = np.arange(self.dimension)[self.vertex_degrees == 1]
            assert len(endpoints) == 2
            endpoint1, endpoint2 = sorted(endpoints)
            self.feasible_edges[endpoint1, endpoint2] = True

    def cost(self):
        return np.sum(self.segments * self.problem.costs, axis=(0, 1))

    def show(self):
        self.problem.show(self)


@pytest.fixture
def berlin_problem():
    '''
    52 locations in Berlin.

    From http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/index.html
    '''
    return TSPProblem("berlin52.tsp")


@pytest.fixture
def berlin_solution(berlin_problem):
    '''
    Optimal tour for the Berlin problem above.

    From http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/index.html
    '''
    sol = TSPSolution(berlin_problem, "berlin52.opt.tour")
    assert np.sum(sol.segments, axis=(0, 1)) == sol.dimension
    return sol


def test_cost_calculation(berlin_solution):
    '''
    Verify that the cost for the Berlin tour is calculated properly.

    7542 comes from http://elib.zib.de/pub/mp-testdata/tsp/tsplib/stsp-sol.html
    '''
    assert berlin_solution.cost() == 7542


if __name__ == '__main__':
    import sys
    problem = TSPProblem(sys.argv[1])
    solution = TSPSolution(problem, sys.argv[2])
    solution.show()
