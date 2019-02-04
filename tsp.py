import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import optimizers
import order_problem

from collections import defaultdict

FIGSIZE = 6


class TSPProblem(order_problem.OrderingProblem):
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
            super().__init__(int(self.info['DIMENSION'][0]))

            self.point_set = np.zeros((self.dimension, 2))
            for i in range(1, self.dimension+1):
                line = infile.readline().rstrip()
                # Unfortunately the .tsp files don't distinguish between
                # north vs south latitude or east vs west longitude.
                # So the TSP may be flipped in some cases.
                node_num, latitude, longitude = line.split(' ')
                assert int(node_num) == i
                self.point_set[i-1] = np.array(
                    [float(longitude), float(latitude)]
                )
            line = infile.readline().rstrip()
            assert line == '' or line == 'EOF'

        for i in range(self.dimension):
            for j in range(i+1, self.dimension):
                self.costs[i, j] = round(
                    np.linalg.norm(self.point_set[i]-self.point_set[j])
                )

    def show(self, tour=None):
        point_xs = self.point_set[:, 0]
        point_ys = self.point_set[:, 1]
        plt.figure(figsize=(FIGSIZE, FIGSIZE))
        plt.plot(
            point_xs, point_ys, 'ro', markersize=3,
            scalex=True, scaley=True
        )

        if tour:
            for itinerary in tour.components():
                itinerary_xs = point_xs[itinerary]
                itinerary_ys = point_ys[itinerary]
                plt.plot(
                    itinerary_xs, itinerary_ys, 'kx-', markersize=1
                )

        plt.show()


class TSPSolution(order_problem.OrderingSolution):
    def __init__(self, problem, filepath=None):
        super().__init__(problem)

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
                first_node = int(line) - 1
                latest_node = first_node
                for i in range(self.dimension - 1):
                    line = infile.readline().rstrip()
                    this_node = int(line) - 1
                    self.add_edge(latest_node, this_node)
                    latest_node = this_node
                line = infile.readline().rstrip()
                if line == '-1':
                    self.add_edge(latest_node, first_node)
                    line = infile.readline().rstrip()
                assert line == '' or line == 'EOF'

        self.ensure_validity()

    def ensure_completion(self):
        super().ensure_completion()
        assert np.sum(self.edges_added, axis=(0, 1)) == self.dimension
        assert all(self.node_degrees == 2)

    def add_edge(self, node_a, node_b):
        super().add_edge(node_a, node_b)

        # If the tour is almost complete, make the loop-closing edge feasible.
        if not self.complete and not self.feasible_edges.any():
            endpoints = self.endpoints()
            assert len(endpoints) == 2
            endpoint1, endpoint2 = sorted(endpoints)
            self.feasible_edges[endpoint1, endpoint2] = True

        self.ensure_validity()

    def cost(self):
        self.ensure_validity()
        return np.sum(self.edges_added * self.problem.costs, axis=(0, 1))

    def show(self):
        self.problem.show(self)


TSPProblem.solution_type = TSPSolution


@pytest.fixture
def berlin_problem():
    '''
    52 locations in Berlin.

    From http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/index.html
    '''
    return TSPProblem("berlin52.tsp")


def test_greedy(berlin_problem):
    '''
    Verify the greedy solver runs without errors on the Berlin problem.
    '''
    soln = optimizers.greedy(berlin_problem)
    soln.ensure_completion()
    assert soln.cost() == 9951


@pytest.fixture
def berlin_opt_soln(berlin_problem):
    '''
    Optimal tour for the Berlin problem above.

    From http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/index.html
    '''
    soln = TSPSolution(berlin_problem, "berlin52.opt.tour")
    soln.ensure_completion()
    return soln


def test_cost_calculation(berlin_opt_soln):
    '''
    Verify the cost for the optimal Berlin tour is calculated properly.

    7542 comes from http://elib.zib.de/pub/mp-testdata/tsp/tsplib/stsp-sol.html
    '''
    assert berlin_opt_soln.cost() == 7542


if __name__ == '__main__':
    import sys
    problem = TSPProblem(sys.argv[1])
    if len(sys.argv) <= 2:
        soln = optimizers.genetic(problem, 20, 20)
        print(soln.cost())
        soln.show()
    else:
        soln = TSPSolution(problem, sys.argv[2])
        print(soln.cost())
        soln.show()
