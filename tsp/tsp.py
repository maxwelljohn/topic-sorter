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
        xs = np.array([node[0] for node in self.nodes])
        ys = np.array([node[1] for node in self.nodes])

        if tour:
            xs = xs[tour.nodes]
            ys = ys[tour.nodes]

        plt.figure(figsize=(FIGSIZE, FIGSIZE))
        plt.plot(
            xs, ys, 'ko-' if tour else 'ro', markersize=1,
            scalex=True, scaley=True
        )
        plt.show()


class TSPSolution(object):
    def __init__(self, problem, filepath=None):
        self.problem = problem
        self.segments = np.zeros(
            (self.problem.dimension, self.problem.dimension)
        )

        if filepath:
            assert filepath.endswith('.tour')
            with open(filepath, 'r') as infile:
                self.info = defaultdict(list)
                line = infile.readline()
                while not line.startswith('TOUR_SECTION'):
                    parts = line.split(':')
                    self.info[parts[0].strip()].append(':'.join(parts[1:]).strip())
                    line = infile.readline()
                assert self.info['TYPE'] == ['TOUR']
                assert len(self.info['DIMENSION']) == 1
                self.dimension = int(self.info['DIMENSION'][0])
                assert self.dimension == self.problem.dimension

                self.nodes = []
                for i in range(self.dimension):
                    line = infile.readline().rstrip()
                    # .tour files are 1-indexed, but we 0-index nodes.
                    this_node = int(line)-1
                    if self.nodes:
                        if self.nodes[-1] < this_node:
                            self.segments[self.nodes[-1], this_node] = 1
                        else:
                            self.segments[this_node, self.nodes[-1]] = 1
                    self.nodes.append(this_node)
                line = infile.readline().rstrip()
                if line == '-1':
                    this_node = self.nodes[0]
                    if self.nodes[-1] < this_node:
                        self.segments[self.nodes[-1], this_node] = 1
                    else:
                        self.segments[this_node, self.nodes[-1]] = 1
                    self.nodes.append(this_node)
                    line = infile.readline().rstrip()
                assert line == '' or line == 'EOF'

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
